import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from ..backbone.unimatch.geometry import points_grid
from .ldm_unet.unet import UNetModel
from torch import Tensor
from jaxtyping import Float
from src.global_cfg import get_cfg


def warp_with_pose_depth_candidates(
    feature1,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        points = points_grid(
            b, h, w, device=depth.device
        ).to(pose.dtype)  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = points.view(b, 3, -1)  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = points / points.norm(p=2, dim=1, keepdim=True).clamp(
            min=clamp_min_depth
        )  # normalize
        phi = torch.atan2(points[:, 0], points[:, 2])
        theta = torch.asin(points[:, 1])
        u = (phi + np.pi) / (2 * np.pi)
        v = (theta + np.pi / 2) / np.pi

        # normalize to [-1, 1]
        x_grid = 2 * u - 1
        y_grid = 2 * v - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature


def prepare_feat_proj_data_lists(
    features: Float[Tensor, "b v c h w"],
    extrinsics: Float[Tensor, "b v 4 4"],
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        cur_ref_pose_to_v0_list = []
        for v0, v1 in zip(init_view_order, cur_view_order):
            cur_ref_pose_to_v0_list.append(
                extrinsics[:, v1].clone().detach().float().inverse().type_as(extrinsics)
                @ extrinsics[:, v0].clone().detach()
            )
        cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
        pose_curr_lists.append(cur_ref_pose_to_v0s)

    return feat_lists, pose_curr_lists


class DepthPredictorCascaded(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        mvs_stages,
        feature_channels_list=[128],
        num_depth_candidates_list=[32],
        costvolume_unet_feat_dims_list=[128],
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        num_views=2,
        wo_gbp=True,
    ):
        super(DepthPredictorCascaded, self).__init__()
        self.num_depth_candidates_list = num_depth_candidates_list[:mvs_stages]
        self.costvolume_unet_feat_dims_list = costvolume_unet_feat_dims_list[:mvs_stages]
        self.mvs_stages = mvs_stages
        self.wo_gbp = wo_gbp

        # Cost volume refinement: 2D U-Net
        self.corr_refine_nets = nn.ModuleList()
        self.regressor_residuals = nn.ModuleList()
        self.depth_heads = nn.ModuleList()
        for stage, num_depth_candidate in enumerate(self.num_depth_candidates_list):
            feature_channels = feature_channels_list[stage]
            input_channels = num_depth_candidate + feature_channels
            if stage > 0:
                input_channels += 1  # add 1 for the previous depth
            if not get_cfg().model.encoder.wo_mono_depth:
                input_channels += 32
            channels = self.costvolume_unet_feat_dims_list[stage]
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ]
            self.corr_refine_nets.append(nn.Sequential(*modules))
            # cost volume u-net skip connection
            self.regressor_residuals.append(nn.Conv2d(
                input_channels, channels, 1, 1, 0
            ))

            # Depth estimation: project features to get softmax based coarse depth
            self.depth_heads.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, num_depth_candidate, 3, 1, 1),
            ))

    def forward(
        self,
        inputs,
        extrinsics,
        near,
        far,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""
        outputs = {}
        outputs["stages"] = []
        coarse_disps = None
        features_list = inputs["backbone"]['trans_features']

        for stage_idx in range(self.mvs_stages):
            features = features_list[stage_idx]
            num_depth_candidates = self.num_depth_candidates_list[stage_idx]
            corr_refine_net = self.corr_refine_nets[stage_idx]
            regressor_residual = self.regressor_residuals[stage_idx]
            depth_head = self.depth_heads[stage_idx]

            # format the input
            b, v, c, h, w = features.shape
            feat_comb_lists, pose_curr_lists = prepare_feat_proj_data_lists(
                features, extrinsics
            )

            # prepare depth bound (inverse depth) [v*b, d]
            if stage_idx == 0:
                min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
                max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
                disp_candi_curr = (
                    min_depth
                    + torch.linspace(0.0, 1.0, num_depth_candidates).unsqueeze(0).to(min_depth.device)
                    * (max_depth - min_depth)
                ).type_as(features)
                disp_candi_curr = repeat(disp_candi_curr, "vb d -> vb d h w", h=h, w=w)
                disp_interval = (max_depth - min_depth) / (num_depth_candidates - 1)
            else:
                disp_interval = disp_interval / 2
                shift = torch.linspace(-num_depth_candidates // 2, num_depth_candidates // 2 - 1, num_depth_candidates).type_as(features)
                shift = repeat(shift, "d -> vb d h w", vb=v * b, h=h, w=w)
                disp_curr = F.interpolate(coarse_disps, scale_factor=2, mode='bilinear', align_corners=True).detach()
                disp_candi_curr = (
                    disp_curr + disp_interval.view(disp_curr.shape[0], 1, 1, 1) * shift
                ).clamp(min=rearrange(min_depth, "vb 1 -> vb 1 1 1"), max=rearrange(max_depth, "vb 1 -> vb 1 1 1")) # [VB, D, H, W]

            # cost volume constructions
            feat01 = feat_comb_lists[0]
            raw_correlation_in_lists = []
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                # sample feat01 from feat10 via camera projection
                feat01_warped = warp_with_pose_depth_candidates(
                    feat10,
                    pose_curr,
                    1.0 / disp_candi_curr,
                    warp_padding_mode="zeros",
                )  # [vB, C, D, H, W]
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                    1
                ) / (
                    c**0.5
                )  # [vB, D, H, W]
                raw_correlation_in_lists.append(raw_correlation_in)
            # average all cost volumes
            raw_correlation_in = torch.mean(
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
            )  # [vxb d, h, w]
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
            if stage_idx >= 1:
                raw_correlation_in = torch.cat((raw_correlation_in, disp_curr), dim=1)
            if 'mono_depth' in inputs:
                mono_features = inputs['mono_depth']['mono_feat']
                mono_features = rearrange(mono_features, "b v c h w -> (v b) c h w")
                mono_features = F.interpolate(mono_features, size=raw_correlation_in.shape[-2:], mode='bilinear', align_corners=False)
                raw_correlation_in = torch.cat([raw_correlation_in, mono_features], dim=1)

            # refine cost volume via 2D u-net
            raw_correlation = corr_refine_net(raw_correlation_in)  # (vb d h w)
            # apply skip connection
            raw_correlation = raw_correlation + regressor_residual(
                raw_correlation_in
            )
            raw_correlation = depth_head(raw_correlation)

            # softmax to get coarse depth and density
            pdf = F.softmax(
                raw_correlation, dim=1
            )  # [2xB, D, H, W]
            coarse_disps = (disp_candi_curr * pdf).sum(
                dim=1, keepdim=True
            )  # (vb, 1, h, w)
            pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax

            if not self.wo_gbp and self.training:
                if stage_idx > 0:
                    last_disps = F.interpolate(last_disps, scale_factor=2, mode='bilinear', align_corners=True)
                    residual = coarse_disps - last_disps
                    coarse_disps = (last_disps + residual.detach()) / 2 + coarse_disps / 2
                last_disps = coarse_disps

            outputs_stage = {}
            outputs_stage["features"] = features
            outputs_stage["disp"] = rearrange(coarse_disps, "(v b) 1 h w -> b v h w", v=v)
            outputs_stage["depth"] = 1 / outputs_stage["disp"]
            outputs_stage["photometric_confidence"] = pdf_max
            outputs_stage["raw_correlation"] = rearrange(raw_correlation, "(v b) ... -> b v ...", v=v)
            outputs_stage["disp_candi_curr"] = rearrange(disp_candi_curr, "(v b) ... -> b v ...", v=v)
            outputs["stages"].append(outputs_stage)

        outputs.update(outputs_stage)

        for stage_idx in range(self.mvs_stages, len(features_list)):
            outputs_stage = {}
            outputs_stage['features'] = features_list[stage_idx]
            outputs["stages"].append(outputs_stage)

        return outputs

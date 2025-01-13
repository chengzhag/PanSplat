from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from collections import OrderedDict

from .backbone import (
    BackboneCascaded,
)
from .encoder import Encoder
from .costvolume.depth_predictor_cascaded import DepthPredictorCascaded
from .visualization.encoder_visualizer_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg
from .gaussian_head import GaussianHead
from torch.nn import functional as F
from .unifuse.networks.convert_module import erp_convert
import os


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderPanSplatCfg:
    name: Literal["pansplat_enc"]
    d_feature: List[int]
    num_depth_candidates: List[int]
    visualizer: EncoderVisualizerCostVolumeCfg
    unimatch_weights_path: str | None
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dims: List[int]
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    wo_backbone_cross_attn: bool
    mvs_stages: int
    fpn_stages: int
    gaussian_head: dict
    wo_gbp: bool
    wo_mono_depth: bool
    mono_depth: dict
    unifuse_pretrained_path: str
    habitat_monodepth_path: str
    use_wrap_padding: bool
    fpn_max_height: int
    freeze_mvs: bool


class EncoderPanSplat(Encoder[EncoderPanSplatCfg]):
    backbone: BackboneCascaded
    depth_predictor:  DepthPredictorCascaded

    def __init__(self, cfg: EncoderPanSplatCfg) -> None:
        super().__init__(cfg)
        self.mvs_only = get_cfg().mvs_only

        self.backbone = BackboneCascaded(
            feature_channels=cfg.d_feature[:cfg.fpn_stages],
            no_cross_attn=cfg.wo_backbone_cross_attn,
        )
        self.backbone = erp_convert(self.backbone)
        if get_cfg().mode == 'train':
            ckpt_path = cfg.unimatch_weights_path
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict() and v.shape == self.backbone.state_dict()[k].shape
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=False)

        # monocular depth
        if not cfg.wo_mono_depth:
            from .unifuse.networks import UniFuse

            mono_depth = UniFuse(**cfg.mono_depth)
            if cfg.use_wrap_padding:
                mono_depth.equi_encoder = erp_convert(mono_depth.equi_encoder)
                mono_depth.equi_decoder = erp_convert(mono_depth.equi_decoder)

            ckpt_path = cfg.unifuse_pretrained_path
            if os.path.isfile(ckpt_path):
                print("==> Load pretrained unifuse checkpoint: %s" % ckpt_path)
                unifuse_pretrained_weight = torch.load(ckpt_path)
                replace_dict = {}
                for i, k in enumerate(mono_depth.equi_decoder.keys()):
                    replace_dict[f"equi_decoder.{i}."] = f"equi_decoder.{k}."
                for i, k in enumerate(mono_depth.c2e.keys()):
                    replace_dict[f"projectors.{i}."] = f"c2e.{k}."
                new_unifuse_pretrained_weight = {}
                for k, v in unifuse_pretrained_weight.items():
                    for old_k, new_k in replace_dict.items():
                        k = k.replace(old_k, new_k)
                    new_unifuse_pretrained_weight[k] = v
                mono_depth.load_state_dict(new_unifuse_pretrained_weight, False)
            else:
                print(f"==> No pretrained unifuse checkpoint found at {ckpt_path}")

            # There is a bug in the PanoGRF codebase, where the non erp part of the model has wrong weights
            # So we have to load the weights from the habitat monodepth model
            # The erp part of the model is not overwritten due to disconnect references after erp_convert
            ckpt_path = cfg.habitat_monodepth_path
            if os.path.isfile(ckpt_path):
                print("==> Load finetuned unifuse checkpoint: %s" % ckpt_path)
                habitat_monodepth_weight = torch.load(ckpt_path)["model_state_dict"]
                new_habitat_monodepth_weight = {}
                for k, v in habitat_monodepth_weight.items():
                    for old_k, new_k in replace_dict.items():
                        k = k.replace(old_k, new_k)
                    if 'depthconv_0' in k:
                        continue
                    new_habitat_monodepth_weight[k] = v
                mono_depth.load_state_dict(new_habitat_monodepth_weight, False)
            else:
                print(f"==> No finetuned unifuse checkpoint found at {ckpt_path}")

            mono_depth.requires_grad_(False)
            mono_depth.eval()
            self.mono_depth = mono_depth

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorCascaded(
            mvs_stages=cfg.mvs_stages,
            feature_channels_list=cfg.d_feature[:cfg.fpn_stages],
            num_depth_candidates_list=cfg.num_depth_candidates[:cfg.fpn_stages],
            costvolume_unet_feat_dims_list=cfg.costvolume_unet_feat_dims[:cfg.fpn_stages],
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            wo_gbp=cfg.wo_gbp,
        )
        if cfg.use_wrap_padding:
            self.depth_predictor = erp_convert(self.depth_predictor)

        # gaussians convertor
        if not self.mvs_only:
            if not isinstance(cfg.gaussian_head['opacity_mapping'], OpacityMappingCfg):
                cfg.gaussian_head['opacity_mapping'] = OpacityMappingCfg(
                    **cfg.gaussian_head['opacity_mapping']
                )
            self.gaussian_head = GaussianHead(
                **cfg.gaussian_head,
                image_height=get_cfg().dataset.image_shape[0],
                encoder_cfg=cfg,
            )

        if cfg.freeze_mvs:
            for param in self.depth_predictor.parameters():
                param.requires_grad = False
            for param in self.backbone.parameters():
                param.requires_grad = False

    def mvs_forward(
        self,
        context: dict,
        visualization_dump: Optional[dict] = None,
    ) -> dict:
        b, v, _, h, w = context["image"].shape
        outputs = {}

        # Encode the context images.
        mvs_image = context["image"]
        if h > self.cfg.fpn_max_height:
            mvs_image = rearrange(mvs_image, "b v c h w -> (b v) c h w")
            mvs_image = F.interpolate(
                mvs_image,
                size=(self.cfg.fpn_max_height, self.cfg.fpn_max_height * 2),
                mode="bilinear"
            )
            mvs_image = rearrange(mvs_image, "(b v) c h w -> b v c h w", b=b)
        outputs["backbone"] = self.backbone(
            mvs_image,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
        )

        # Monocular depth
        if not self.cfg.wo_mono_depth:
            mono_erp_inputs = rearrange(context["mono_image"], "b v c h w -> (b v) c h w")
            mono_cube_inputs = rearrange(context["cube_image"], "b v c h (f w) -> (b v) c h (f w)", f=2)
            mono_depth = self.mono_depth(mono_erp_inputs, mono_cube_inputs)
            mono_depth["pred_depth"] = rearrange(mono_depth["pred_depth"], "(b v) 1 h w -> b v h w", b=b)
            mono_depth["mono_feat"] = rearrange(mono_depth["mono_feat"], "(b v) c h w -> b v c h w", b=b)
            outputs["mono_depth"] = mono_depth

            if visualization_dump is not None:
                visualization_dump["mono_depth"] = mono_depth["pred_depth"]

        # Sample depths from the resulting features.
        mvs_outputs = self.depth_predictor(
            outputs,
            context["extrinsics"],
            context["near"],
            context["far"],
        )

        if visualization_dump is not None:
            visualization_dump["depth_wo_refine"] = [
                stage['depth'] for stage in mvs_outputs['stages'] if 'depth' in stage
            ]
        outputs["mvs_outputs"] = mvs_outputs

        return outputs

    def gh_forward(
        self,
        mvs_outputs,
        context,
        global_step=None,
        inference=False,
    ):
        gaussians = self.gaussian_head(
            mvs_outputs,
            context,
            global_step,
            inference,
        )

        # # Dump visualizations if needed.
        # if visualization_dump is not None:
        #     visualization_dump["depth"] = gaussians['depths']
        #     visualization_dump["scales"] = gaussians["gaussians_vis"].scales
        #     visualization_dump["rotations"] = gaussians["gaussians_vis"].rotations

        return gaussians

    def forward(
        self,
        context: dict,
        global_step: int = None,
        visualization_dump: Optional[dict] = None,
        inference: bool = False
    ) -> dict:
        outputs = self.mvs_forward(
            context,
            visualization_dump,
        )

        if not self.mvs_only:
            outputs["gaussians"] = self.gh_forward(
                outputs["mvs_outputs"],
                context,
                global_step,
                inference,
            )

        return outputs

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None

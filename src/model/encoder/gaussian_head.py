import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter

import torch.nn.functional as F
import numpy as np
import math
from src.global_cfg import get_cfg
from dataclasses import fields


class GaussianHead(nn.Module):

    def __init__(
        self,
        image_height,
        gaussian_adapter,
        gaussians_per_pixel,
        num_surfaces,
        opacity_mapping,
        wo_pgs,
        wo_pgs_res,
        gh_mvs_scale_factor,
        unify_gh_res,
        wo_disp_dens_refine,
        wo_fibo_gs,
        wo_sh_res,
        wo_feat,
        fibo_mlp_layers,
        gh_cnn_layers,
        patchs_height,
        patchs_width,
        encoder_cfg,
        deferred_blend,
    ):
        super().__init__()
        # Configs from the encoder
        feature_channels_list = encoder_cfg.d_feature[:encoder_cfg.fpn_stages]
        mvs_stages = encoder_cfg.mvs_stages
        num_depth_candidates = encoder_cfg.num_depth_candidates[min(encoder_cfg.mvs_stages, encoder_cfg.fpn_stages) - 1]

        last_stage_height = image_height
        last_gh_stage = 0
        if gh_mvs_scale_factor is not None:
            last_stage_height = min(last_stage_height, encoder_cfg.fpn_max_height)
            last_gh_stage = len(feature_channels_list) - mvs_stages - round(math.log2(gh_mvs_scale_factor))
            feature_channels_list = feature_channels_list[:mvs_stages]
        last_scale = 2**last_gh_stage
        last_stage_height = last_stage_height // last_scale

        if wo_pgs:
            feature_channels_list = [feature_channels_list[-1]]
        self.feature_channels_list = feature_channels_list

        assert gaussians_per_pixel == 1, "Only support gaussians_per_pixel=1"
        self.gaussians_per_pixel = gaussians_per_pixel
        self.num_surfaces = num_surfaces
        self.gaussian_adapter = GaussianAdapter(gaussian_adapter)
        self.gaussian_raw_channels = num_surfaces * (self.gaussian_adapter.d_in + 2)
        self.opacity_mapping = opacity_mapping
        self.wo_pgs = wo_pgs
        self.wo_pgs_res = wo_pgs_res
        self.wo_disp_dens_refine = wo_disp_dens_refine
        self.wo_fibo_gs = wo_fibo_gs
        self.wo_sh_res = wo_sh_res
        self.wo_feat = wo_feat
        self.num_patchs = patchs_height * patchs_width
        assert self.num_patchs > 0, "num_patchs should be greater than 0"
        self.padding = gh_cnn_layers
        self.deferred_blend = deferred_blend

        self.to_gaussians_list = nn.ModuleList()
        if not wo_fibo_gs:
            self.gaussians_mlp_list = nn.ModuleList()

        self.full_shape = []
        self.gh_stages = len(feature_channels_list)
        for stage_idx, feature_channels in enumerate(feature_channels_list):
            # Stage shape
            scale = 2**(self.gh_stages - stage_idx - 1)
            stage_height = last_stage_height if unify_gh_res else last_stage_height // scale
            self.full_shape.append((stage_height, stage_height * 2))

            # Gaussians xy and patch range
            patch_info = self.stage_patch_info(stage_idx, patchs_height, patchs_width)
            for patch_idx, (gs_xy_patch, range_xy_patch, range_hw_patch) in enumerate(zip(*patch_info)):
                key = f"{stage_idx}_{patch_idx}"
                self.register_buffer(f"gs_xy_{key}", gs_xy_patch, persistent=False)
                self.register_buffer(f"range_hw_{key}", range_hw_patch, persistent=False)
                self.register_buffer(f"range_xy_{key}", range_xy_patch, persistent=False)

            # Gaussians prediction: covariance, color
            gau_in = 3 + num_depth_candidates
            if not wo_feat:
                gau_in += feature_channels
            gau_out = self.gaussian_raw_channels
            if not wo_disp_dens_refine:
                gau_out += gaussians_per_pixel + gaussians_per_pixel * num_depth_candidates
            gau_hid = gau_out * 2
            if stage_idx > 0 and not wo_pgs_res:
                gau_in += gau_out if wo_fibo_gs else gau_hid
            self.to_gaussians_list.append(self.gaussians_cnn(
                gau_in, gau_hid, gau_out if wo_fibo_gs else gau_hid, gh_cnn_layers
            ))
            if not wo_fibo_gs:
                self.gaussians_mlp_list.append(self.fibo_mlp(
                    gau_hid, gau_hid, gau_out, fibo_mlp_layers
                ))

    def stage_patch_info(self, stage_idx, patchs_height, patchs_width):
        h, w = self.full_shape[stage_idx]
        patch_h, patch_w = h // patchs_height, w // patchs_width

        range_hw = []
        for i in range(patchs_height):
            for j in range(patchs_width):
                h_start = i * patch_h
                h_end = (i + 1) * patch_h
                w_start = j * patch_w
                w_end = (j + 1) * patch_w
                patch_hw = torch.tensor([[h_start, h_end], [w_start, w_end]])
                range_hw.append(patch_hw)

        gs_xy = []
        if self.wo_fibo_gs:
            xy, _ = sample_image_grid((h, w))
            xy = xy * 2 - 1
            xy = rearrange(xy, "(nh h) (nw w) xy -> (nh nw) (h w) xy", nh=patchs_height, nw=patchs_width)
            gs_xy.extend(list(xy))
        else:
            lonlat = fibonacci_sphere_grid(c=w)
            xy = lonlat / lonlat.new_tensor([np.pi, np.pi / 2])
            for patch_hw in range_hw:
                patch_xy = patch_hw.flip(0).float()
                patch_xy = patch_xy / patch_xy.new_tensor([w, h]).unsqueeze(-1) * 2 - 1
                start, end = patch_xy.unbind(1)
                eps = 1e-6
                start[start <= -1 + eps] = -1 - eps
                end[end >= 1 - eps] = 1 + eps
                xy_patch = xy[(xy[:, 0] >= start[0]) & (xy[:, 0] < end[0]) & (xy[:, 1] >= start[1]) & (xy[:, 1] < end[1])]
                gs_xy.append(xy_patch)

        range_xy = []
        for patch_hw in range_hw:
            patch_hw[:, 1] += self.padding
            patch_hw[:, 0] -= self.padding
            patch_xy = patch_hw.flip(0).float()
            patch_xy = patch_xy / patch_xy.new_tensor([w, h]).unsqueeze(-1) * 2 - 1
            range_xy.append(patch_xy)

        return gs_xy, range_xy, range_hw

    def gaussians_cnn(self, in_channels, hidden_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden_channels, 3, 1, 1))
            layers.append(nn.GELU())
            in_channels = hidden_channels
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def fibo_mlp(self, in_channels, hidden_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.GELU())
            in_channels = hidden_channels
        layers.append(nn.Linear(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, mvs_outputs, context, global_step=None, inference=False):
        b, v, c, _, _ = context['image'].shape

        # forward for each patch
        gaussians_patchs = []
        gaussians_patch_idx = []
        gaussians_stages_patchs = []
        for patch_idx in range(self.num_patchs):
            patch = self.patch_forward(mvs_outputs, context, global_step, patch_idx)
            gaussians_patchs.append(patch["gaussians"])
            if self.deferred_blend:
                gaussians_patch_idx.append([
                    p.opacities.new_full(
                        p.opacities.shape, patch_idx
                    ) for p in patch["gaussians"]
                ])
            else:
                gaussians_patch_idx.append(patch["gaussians"].opacities.new_full(
                    patch["gaussians"].opacities.shape, patch_idx))
            gaussians_stages_patchs.append(patch["stages"])
        self.clean_padded_cache()

        # merge the gaussians
        if self.deferred_blend:
            gaussians = []
            gaussians_idx = []
            for i in range(v):
                gaussians.append(Gaussians.from_list([p[i] for p in gaussians_patchs]))
                gaussians_idx.append(torch.cat([p[i] for p in gaussians_patch_idx], dim=1))

            outputs = {
                "gaussians": gaussians,
                "patch_idx": gaussians_idx,
            }
        else:
            outputs = {
                "gaussians": Gaussians.from_list(gaussians_patchs),
                "patch_idx": torch.cat(gaussians_patch_idx, dim=1),
            }

        if not inference:
            gaussians_stages = []
            for stage_idx in range(len(gaussians_stages_patchs[0])):
                if self.deferred_blend:
                    gaussians_stage = [
                        Gaussians.from_list([p[stage_idx]["gaussians"][i] for p in gaussians_stages_patchs])
                        for i in range(v)
                    ]
                else:
                    gaussians_stage = [p[stage_idx]["gaussians"] for p in gaussians_stages_patchs]
                    gaussians_stage = Gaussians.from_list(gaussians_stage)
                gaussians_stages.append({"gaussians": gaussians_stage})
            outputs["stages"] = gaussians_stages

        return outputs

    def clean_padded_cache(self):
        # must be called after forward
        self.padded_cache = [{} for _ in range(self.gh_stages)]

    def cache_padding(self, f, stage_idx, key):
        if not hasattr(self, "padded_cache"):
            self.clean_padded_cache()
        stage = self.padded_cache[stage_idx]
        if key in stage:
            return stage[key]
        full = F.interpolate(f, size=self.full_shape[stage_idx], mode="bilinear")
        full = pad_pano(full, self.padding)
        stage[key] = full
        return full

    def patch_forward(self, outputs, context, global_step=None, patch_idx=0):
        b, v, c, _, _ = context['image'].shape
        images_fullres = rearrange(context['image'], "b v c h w -> (v b) c h w")
        raw_correlation_fullres = rearrange(outputs["raw_correlation"], "b v ... -> (v b) ...")
        disp_candi_curr_fullres = rearrange(outputs["disp_candi_curr"], "b v ... -> (v b) ...", v=v, b=b)
        features_full = None
        disp_full = None
        pdf_max_full = None

        stages = outputs["stages"]
        # if wo pyramid gaussians, only use the last stage
        if self.wo_pgs:
            stages = [stages[-1]]
        # limit the number of stages
        stages = stages[:self.gh_stages]

        gaussians = {}
        gaussians["stages"] = []
        for stage_idx, stage in enumerate(stages):
            # crop inputs
            if not self.wo_feat:
                features_full = stage.get("features", features_full)
                features = rearrange(features_full, "b v ... -> (v b) ...")
                features = self.crop_patch(features, stage_idx, patch_idx, "features")
            images = self.crop_patch(images_fullres, stage_idx, patch_idx, "images")
            raw_correlation = self.crop_patch(raw_correlation_fullres, stage_idx, patch_idx, "raw_correlation")

            # fibonnaci sphere grid
            xy = getattr(self, f"gs_xy_{stage_idx}_{patch_idx}")
            full_grid = repeat(xy, "n xy -> vb n 1 xy", vb=v * b)

            # gaussians head
            raw_gaussians_in = [images, raw_correlation]
            if not self.wo_feat:
                raw_gaussians_in.insert(1, features)
            raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)

            # add residual
            if stage_idx > 0 and not self.wo_pgs_res:
                last_raw_gaussians = F.interpolate(
                    last_raw_gaussians, scale_factor=2, mode="bilinear")
                last_raw_gaussians = unpad_pano(last_raw_gaussians, self.padding)
                raw_gaussians_in = torch.cat([raw_gaussians_in, last_raw_gaussians], dim=1)

            delta_raw_gaussians = self.to_gaussians_list[stage_idx](raw_gaussians_in)

            # add residual
            if stage_idx == 0 or self.wo_pgs_res:
                raw_gaussians = delta_raw_gaussians
            else:
                raw_gaussians = last_raw_gaussians + delta_raw_gaussians
            last_raw_gaussians = raw_gaussians

            # gaussians prediction
            sh_res = (stage_idx == 0 and not self.wo_sh_res) or (not self.wo_sh_res and self.wo_pgs_res)
            if self.wo_fibo_gs:
                disp_candi_curr = self.crop_patch(disp_candi_curr_fullres, stage_idx, patch_idx, "disp_candi_curr")
                disp_candi_curr = unpad_pano(disp_candi_curr, self.padding)
                disp_candi_curr = rearrange(disp_candi_curr, "(v b) d h w -> b v (h w) d", v=v, b=b)
                raw_gaussians = unpad_pano(raw_gaussians, self.padding)
                raw_gaussians = rearrange(raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b)
                if sh_res:
                    sh_base = unpad_pano(images, self.padding)
                    sh_base = rearrange(sh_base, "(v b) c h w -> b v (h w) c", v=v, b=b)
            else:
                disp_candi_curr = F.grid_sample(disp_candi_curr_fullres, full_grid, padding_mode="border")
                disp_candi_curr = rearrange(disp_candi_curr, "(v b) d n 1 -> b v n d", v=v, b=b)
                patch_grid = repeat(self.map_patch_xy(xy, stage_idx, patch_idx), "n xy -> vb n 1 xy", vb=v * b)
                raw_gaussians = F.grid_sample(raw_gaussians, patch_grid, padding_mode="border")
                raw_gaussians = rearrange(raw_gaussians, "(v b) c n 1 -> b v n c", v=v, b=b)
                raw_gaussians = self.gaussians_mlp_list[stage_idx](raw_gaussians)
                if sh_res:
                    sh_base = F.grid_sample(images, patch_grid, padding_mode="border")
                    sh_base = rearrange(sh_base, "(v b) c n 1 -> b v n c", v=v, b=b)

            # disp and density prediction
            if not self.wo_disp_dens_refine:
                raw_gaussians, raw_correlation, raw_density = raw_gaussians.split(
                    [
                        self.gaussian_raw_channels,
                        self.gaussians_per_pixel * raw_correlation_fullres.shape[1],
                        self.gaussians_per_pixel
                    ], dim=-1
                )
                pdf = F.softmax(raw_correlation, dim=-1)
                fine_disp = (disp_candi_curr * pdf).sum(dim=-1, keepdim=True)
            else:
                disp_full = stage.get("disp", disp_full)
                disp = rearrange(disp_full, "b v ... -> (v b) () ...")
                disp = self.crop_patch(disp, stage_idx, patch_idx, "disp")
                pdf_max_full = stage.get("photometric_confidence", pdf_max_full)
                pdf_max = rearrange(pdf_max_full, "b v ... -> (v b) () ...")
                pdf_max = self.crop_patch(pdf_max, stage_idx, patch_idx, "pdf_max")
                if self.wo_fibo_gs:
                    pdf_max = unpad_pano(pdf_max, self.padding)
                    pdf_max = rearrange(pdf_max, "(v b) c h w -> b v (h w) c", v=v, b=b)
                    disp = unpad_pano(disp, self.padding)
                    disp = rearrange(disp, "(v b) c h w -> b v (h w) c", v=v, b=b)
                else:
                    pdf_max = F.grid_sample(pdf_max, patch_grid, align_corners=False, padding_mode="border")
                    pdf_max = rearrange(pdf_max, "(v b) c n 1 -> b v n c", v=v, b=b)
                    disp = F.grid_sample(disp, patch_grid, align_corners=False, padding_mode="border")
                    disp = rearrange(disp, "(v b) c n 1 -> b v n c", v=v, b=b)
                raw_density = pdf_max
                fine_disp = disp

            densities = F.sigmoid(raw_density)
            fine_disp = fine_disp.clamp(
                1. / rearrange(context['far'], "b v -> b v () ()"),
                1. / rearrange(context['near'], "b v -> b v () ()"),
            )
            depths = 1.0 / fine_disp

            # convert gaussians
            gaussians_reshape = self.convert_gaussians(
                depths, densities, raw_gaussians, context["extrinsics"],
                xy, self.full_shape[stage_idx], global_step,
                sh_base if sh_res else None
            )
            gaussians_stage = {
                "gaussians": gaussians_reshape,
                # "images": rearrange(images, "(v b) c h w -> b v c h w", v=v, b=b),
            }
            gaussians["stages"].append(gaussians_stage)

        gaussians.update(gaussians_stage)
        if self.deferred_blend:
            gaussians['gaussians'] = [Gaussians.from_list([g["gaussians"][i] for g in gaussians["stages"]]) for i in range(v)]
        else:
            gaussians['gaussians'] = Gaussians.from_list([g["gaussians"] for g in gaussians["stages"]])
        return gaussians

    def crop_patch(self, f, stage_idx, patch_idx, key):
        full = self.cache_padding(f, stage_idx, key)
        range_hw = getattr(self, f"range_hw_{stage_idx}_{patch_idx}")
        range_hw = range_hw + self.padding
        patch = full[..., range_hw[0, 0]:range_hw[0, 1], range_hw[1, 0]:range_hw[1, 1]]
        return patch

    def map_patch_xy(self, xy, stage_idx, patch_idx):
        range_xy = getattr(self, f"range_xy_{stage_idx}_{patch_idx}")
        patch_size = range_xy[:, 1] - range_xy[:, 0]
        xy = (xy - range_xy[:, 0]) / patch_size * 2 - 1
        return xy

    def convert_gaussians(self, depths, densities, raw_gaussians, extrinsics,
                          xy, image_size, global_step=None, sh_base=None):
        h, w = image_size
        device = depths.device

        densities = repeat(densities, "b v n dpt -> b v n srf dpt", srf=self.num_surfaces)
        depths = repeat(depths, "b v n dpt -> b v n srf dpt", srf=self.num_surfaces)

        # Convert the features and depths into Gaussians.
        xy_ray = rearrange(xy, "n xy -> n 1 xy")
        xy_ray = xy_ray / 2 + 0.5

        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()

        pixel_size = 1 / torch.tensor((w, h), device=device).type_as(xy)
        if not self.wo_fibo_gs:
            lat = xy[..., 1] * np.pi / 2
            r = torch.cos(lat)
            r[r < 1e-2] = 1e-2
            pixel_width = 1 / r
            pixel_height = pixel_width.new_ones(pixel_width.shape)
            pixel_size = torch.stack((pixel_width, pixel_height), dim=-1) * pixel_size
            pixel_size = rearrange(pixel_size, "n xy -> n 1 xy")
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            sh_base=rearrange(
                sh_base,
                "b v n c -> b v n () () c"
            ) if sh_base is not None else None,
        )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1
        gaussians.opacities = opacity_multiplier * gaussians.opacities

        keys = [f.name for f in fields(Gaussians)]
        if self.deferred_blend:
            kwargs = [{
                k: rearrange(
                    getattr(gaussians, k)[:, v],
                    "b r srf spp ... -> b (r srf spp) ...",
                ) for k in keys
            } for v in range(gaussians.means.shape[1])]
            gaussians_reshape = [Gaussians(**kw) for kw in kwargs]
        else:
            kwargs = {
                k: rearrange(
                    getattr(gaussians, k),
                    "b v r srf spp ... -> b (v r srf spp) ...",
                ) for k in keys
            }
            gaussians_reshape = Gaussians(**kwargs)
        return gaussians_reshape

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int | None = None,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.opacity_mapping
        if global_step is None:
            exponent = 1
        else:
            x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
            exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))


def pad_pano(pano, padding):
    if padding <= 0:
        return pano

    if pano.ndim == 5:
        b, m = pano.shape[:2]
        pano_pad = rearrange(pano, 'b m c h w -> (b m c) h w')
    elif pano.ndim == 4:
        b = pano.shape[0]
        pano_pad = rearrange(pano, 'b c h w -> (b c) h w')
    else:
        raise NotImplementedError('pano should be 4 or 5 dim')

    pano_pad = F.pad(pano_pad, [padding, ] * 2, mode='circular')
    pano_pad = F.pad(pano_pad, [0, 0, padding, padding], mode='constant')

    if pano.ndim == 5:
        pano_pad = rearrange(pano_pad, '(b m c) h w -> b m c h w', b=b, m=m)
    elif pano.ndim == 4:
        pano_pad = rearrange(pano_pad, '(b c) h w -> b c h w', b=b)

    return pano_pad


def unpad_pano(pano_pad, padding):
    if padding <= 0:
        return pano_pad
    return pano_pad[..., padding:-padding, padding:-padding]


# The fibonacci_sphere function is from from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples=1000):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = torch.linspace(1, -1, samples)
    radius = np.sqrt(1 - y**2)
    theta = phi * torch.arange(samples)
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    return torch.stack((x, y, z), dim=1)


# Generate grid_sample points on the equirectangular projection
def fibonacci_sphere_grid(c, device=None):
    r = c / 2 / np.pi
    samples = 4 * np.pi * r**2
    samples = int(samples)
    points = fibonacci_sphere(samples).to(device)

    x, y, z = points.T
    lon = torch.atan2(x, z)
    lat = torch.asin(y)

    return torch.stack((lon, lat), dim=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pathlib import Path

    output_dir = Path("outputs/fibo")
    output_dir.mkdir(parents=True, exist_ok=True)
    height = 8
    width = height * 2
    radius = 1
    size = 100
    elev = 30
    azim = -60

    # draw fibo with equirectangular projection
    lon, lat = fibonacci_sphere_grid(width).T
    plt.scatter(lon, lat, size)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi / 2, np.pi / 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_dir / "fibonacci_lattice.png", bbox_inches='tight')
    plt.savefig(output_dir / "fibonacci_lattice.svg", bbox_inches='tight', format="svg", transparent=True)
    plt.close()

    # draw pixel-aligned grid
    u = np.linspace(
        -np.pi + np.pi / width,
        np.pi - np.pi / width,
        width
    )
    v = np.linspace(
        -np.pi / 2 + np.pi / height / 2,
        np.pi / 2 - np.pi / height / 2,
        height
    )
    uu, vv = np.meshgrid(u, v)
    plt.scatter(uu, vv, s=size)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi / 2, np.pi / 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_dir / "pixel_aligned.png", bbox_inches='tight')
    plt.savefig(output_dir / "pixel_aligned.svg", bbox_inches='tight', format="svg", transparent=True)
    plt.close()

    # draw fibo 3d sphere
    u = np.linspace(0, 2 * np.pi, width + 1)
    v = np.linspace(0, np.pi, height + 1)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    # mask the points behind the surface
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)
    cam_dir_x = np.cos(azim_rad) * np.cos(elev_rad)
    cam_dir_y = np.sin(azim_rad) * np.cos(elev_rad)
    cam_dir_z = np.sin(elev_rad)
    cam_dir = np.array([cam_dir_x, cam_dir_y, cam_dir_z])
    points = np.stack((x, y, z), axis=1)
    depth = np.dot(points, cam_dir)
    mask = depth > 0
    x, y, z = x[mask], y[mask], z[mask]
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='w', alpha=1., edgecolor='k', linewidth=0.2, shade=False)
    ax.scatter(x, y, z, s=size, depthshade=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.01, 1.01])
    ax.set_ylim([-1.01, 1.01])
    ax.set_zlim([-1.01, 1.01])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(output_dir / "fibo_glob.png", bbox_inches='tight')
    plt.savefig(output_dir / "fibo_glob.svg", bbox_inches='tight', format="svg", transparent=True)
    plt.close()

    # draw pixel-aligned 3d sphere
    x = radius * np.cos(vv) * np.cos(uu)
    y = radius * np.cos(vv) * np.sin(uu)
    z = radius * np.sin(vv)
    # mask the points behind the surface
    points = np.stack((x, y, z), axis=2)
    depth = np.dot(points, cam_dir)
    mask = depth > 0
    x, y, z = x[mask], y[mask], z[mask]
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='w', alpha=1., edgecolor='k', linewidth=0.2, shade=False)
    ax.scatter(x, y, z, s=size, depthshade=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.01, 1.01])
    ax.set_ylim([-1.01, 1.01])
    ax.set_zlim([-1.01, 1.01])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(output_dir / "pixel_aligned_glob.png", bbox_inches='tight')
    plt.savefig(output_dir / "pixel_aligned_glob.svg", bbox_inches='tight', format="svg", transparent=True)
    plt.close()

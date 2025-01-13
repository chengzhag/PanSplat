from pathlib import Path
from random import randrange
from typing import Optional, List

import numpy as np
import torch
import wandb
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float
from torch import Tensor

from ....dataset.types import BatchedViews
from ....misc.heterogeneous_pairings import generate_heterogeneous_index
from ....visualization.annotation import add_label
from ....visualization.color_map import apply_color_map, apply_color_map_to_image
from ....visualization.colors import get_distinct_color
from ....visualization.drawing.lines import draw_lines
from ....visualization.drawing.points import draw_points
from ....visualization.layout import add_border, hcat, vcat
# from ...ply_export import export_ply
from ..encoder_pansplat import EncoderPanSplat
from .encoder_visualizer import EncoderVisualizer
from .encoder_visualizer_cfg import EncoderVisualizerCostVolumeCfg


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)


class EncoderVisualizerPanSplat(
    EncoderVisualizer[EncoderVisualizerCostVolumeCfg, EncoderPanSplat]
):
    def visualize(
        self,
        context: BatchedViews,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        visualization_dump = {}

        encoder_output = self.encoder(
            context,
            global_step,
            visualization_dump=visualization_dump,
        )

        # Generate high-resolution context images that can be drawn on.
        context_images = context["image"]
        _, _, _, h, w = context_images.shape
        length = min(h, w)
        min_resolution = self.cfg.min_resolution
        scale_multiplier = (min_resolution + length - 1) // length
        if scale_multiplier > 1:
            context_images = repeat(
                context_images,
                "b v c h w -> b v c (h rh) (w rw)",
                rh=scale_multiplier,
                rw=scale_multiplier,
            )

        visualization = {}
        if "gaussians" in encoder_output:
            result = encoder_output["gaussians"]["gaussians"]

            # This is kind of hacky for now, since we're using it for short experiments.
            if self.cfg.export_ply and wandb.run is not None:
                name = wandb.run._name.split(" ")[0]
                ply_path = Path(f"outputs/gaussians/{name}/{global_step:0>6}.ply")
                export_ply(
                    context["extrinsics"][0, 0],
                    result.means[0],
                    visualization_dump["scales"][0],
                    visualization_dump["rotations"][0],
                    result.harmonics[0],
                    result.opacities[0],
                    ply_path,
                )

            # visualization["gaussians"] = self.visualize_pyramid_gaussians(
            #     encoder_output["gaussians"]
            # )

        if "mono_depth" in visualization_dump:
            visualization["mono_depth"] = self.visualize_single_depth(
                context,
                visualization_dump["mono_depth"],
            )

        if "depth" in visualization_dump:
            visualization["depth"] = self.visualize_depth(
                context,
                visualization_dump["depth"],
            )

        if "depth_wo_refine" in visualization_dump:
            visualization["depth_wo_refine"] = self.visualize_depth_wo_refine(
                context,
                visualization_dump["depth_wo_refine"],
            )

        return visualization

    def visualize_depth(
        self,
        context: BatchedViews,
        multi_depth: Float[Tensor, "batch view height width surface spp"],
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        multi_vis = []
        *_, srf, _ = multi_depth.shape
        for i in range(srf):
            depth = multi_depth[..., i, :]
            depth = depth.mean(dim=-1)

            # Compute relative depth and disparity.
            near = rearrange(context["near"], "b v -> b v () ()")
            far = rearrange(context["far"], "b v -> b v () ()")
            relative_depth = (depth - near) / (far - near)
            relative_disparity = 1 - (1 / depth - 1 / far) / (1 / near - 1 / far)

            relative_depth = apply_color_map_to_image(relative_depth, "turbo")
            relative_depth = vcat(*[hcat(*x) for x in relative_depth])
            relative_depth = add_label(relative_depth, "Depth")
            relative_disparity = apply_color_map_to_image(relative_disparity, "turbo")
            relative_disparity = vcat(*[hcat(*x) for x in relative_disparity])
            relative_disparity = add_label(relative_disparity, "Disparity")
            multi_vis.append(add_border(hcat(relative_depth, relative_disparity)))

        return add_border(vcat(*multi_vis))

    def visualize_single_depth(
        self,
        context: BatchedViews,
        depth,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        near = rearrange(context["near"], "b v -> b v () ()")
        far = rearrange(context["far"], "b v -> b v () ()")
        relative_depth = (depth - near) / (far - near)
        relative_disparity = 1 - (1 / depth - 1 / far) / (1 / near - 1 / far)

        relative_depth = apply_color_map_to_image(relative_depth, "turbo")
        relative_depth = vcat(*[hcat(*x) for x in relative_depth])
        relative_depth = add_label(relative_depth, "Depth")
        relative_disparity = apply_color_map_to_image(relative_disparity, "turbo")
        relative_disparity = vcat(*[hcat(*x) for x in relative_disparity])
        relative_disparity = add_label(relative_disparity, "Disparity")

        row = hcat(relative_depth, relative_disparity)
        return row

    def visualize_depth_wo_refine(
        self,
        context: BatchedViews,
        multi_depth,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        multi_vis = []
        for stage, depth in enumerate(multi_depth):
            row = self.visualize_single_depth(context, depth)
            multi_vis.append(add_label(add_border(row), f"Stage {stage + 1}"))

        return add_border(vcat(*multi_vis))

    def visualize_overlaps(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        is_monocular: Optional[Bool[Tensor, "batch view height width"]] = None,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        device = context_images.device
        b, v, _, h, w = context_images.shape
        green = torch.tensor([0.235, 0.706, 0.294], device=device)[..., None, None]
        rb = randrange(b)
        valid = sampling.valid[rb].float()
        ds = self.encoder.cfg.epipolar_transformer.downscale
        valid = repeat(
            valid,
            "v ov (h w) -> v ov c (h rh) (w rw)",
            c=3,
            h=h // ds,
            w=w // ds,
            rh=ds,
            rw=ds,
        )

        if is_monocular is not None:
            is_monocular = is_monocular[rb].float()
            is_monocular = repeat(is_monocular, "v h w -> v c h w", c=3, h=h, w=w)

        # Select context images in grid.
        context_images = context_images[rb]
        index, _ = generate_heterogeneous_index(v)
        valid = valid * (green + context_images[index]) / 2

        vis = vcat(*(hcat(im, hcat(*v)) for im, v in zip(context_images, valid)))
        vis = add_label(vis, "Context Overlaps")

        if is_monocular is not None:
            vis = hcat(vis, add_label(vcat(*is_monocular), "Monocular?"))

        return add_border(vis)

    def visualize_pyramid_gaussians(
        self,
        gaussians,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        multi_vis = []
        for stage_idx, stage in enumerate(gaussians['stages']):
            g = stage['gaussians_vis']
            row = self.visualize_gaussians(
                stage['images'],
                g.opacities,
                g.covariances,
                g.harmonics[..., 0],
            )
            multi_vis.append(add_label(row, f"Stage {stage_idx + 1}"))
        return add_border(vcat(*multi_vis))

    def visualize_gaussians(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        b, v, _, h, w = context_images.shape
        rb = randrange(b)
        context_images = context_images[rb]
        opacities = repeat(
            opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=h, w=w
        )
        colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        # Color-map Gaussian covariawnces.
        det = covariances[rb].det()
        det = apply_color_map(det / det.max(), "inferno")
        det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        return add_border(
            hcat(
                add_label(box(hcat(*context_images)), "Context"),
                add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
                add_label(
                    box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
                ),
                add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
                add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
            )
        )

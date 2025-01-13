from dataclasses import dataclass
from typing import Literal, List

import torch
from einops import rearrange, repeat, einsum
from jaxtyping import Float, Int
from torch import Tensor
import numpy as np
import torch.nn.functional as F

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    view_batch: int
    super_sampling: float


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

        self.num_faces = 6
        cubemap_Rs = torch.eye(4, dtype=torch.float32)
        cubemap_Rs = repeat(cubemap_Rs, "... -> f ...", f=self.num_faces).clone()
        # 'F', 'R', 'B', 'L', 'U', 'D'
        cubemap_Rs[:, :3, :3] = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            ],
            dtype=torch.float32
        ).inverse()
        self.register_buffer("cubemap_Rs", cubemap_Rs, persistent=False)

        intrinsics = torch.eye(3, dtype=torch.float32)
        fx, fy, cx, cy = 0.5, 0.5, 0.5, 0.5
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        self.register_buffer("intrinsics", intrinsics, persistent=False)

        self.face_w = int(dataset_cfg.image_shape[0] // 2 * cfg.super_sampling)
        tp, coor_x, coor_y = cubmap_mapping(self.face_w, *dataset_cfg.image_shape)
        self.register_buffer("tp", torch.tensor(tp, dtype=torch.float32), persistent=False)
        self.register_buffer("coor_x", torch.tensor(coor_x, dtype=torch.float32), persistent=False)
        self.register_buffer("coor_y", torch.tensor(coor_y, dtype=torch.float32), persistent=False)

        u, v = uv_grid(self.face_w, self.face_w, 'cpu')
        f = 0.5 / torch.tan(torch.deg2rad(u.new_tensor(90.0) / 2))
        u = (u - 0.5) / f
        v = (v - 0.5) / f
        vec = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        depth_norm = torch.norm(vec, dim=-1)
        self.register_buffer("depth_norm", depth_norm, persistent=False)

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cpu: bool = False,
    ):
        b, v, _, _ = extrinsics.shape
        colors = []
        split_size = max(1, self.cfg.view_batch // b)
        for start in range(0, v, split_size):
            end = min(start + split_size, v)
            vb = end - start
            color = render_cuda(
                rearrange(extrinsics[:, start:end], "b vb i j -> (b vb) i j"),
                rearrange(intrinsics[:, start:end], "b vb i j -> (b vb) i j"),
                rearrange(near[:, start:end], "b vb -> (b vb)"),
                rearrange(far[:, start:end], "b vb -> (b vb)"),
                image_shape,
                repeat(self.background_color, "c -> (b vb) c", b=b, vb=vb),
                repeat(gaussians.means, "b g xyz -> (b vb) g xyz", vb=vb),
                repeat(gaussians.covariances, "b g i j -> (b vb) g i j", vb=vb),
                repeat(gaussians.harmonics, "b g c d_sh -> (b vb) g c d_sh", vb=vb),
                repeat(gaussians.opacities, "b g -> (b vb) g", vb=vb),
            )
            color = rearrange(color, "(b vb) c h w -> b vb c h w", b=b, vb=vb)
            if cpu:
                color = color.cpu()
            colors.append(color)
        color = torch.cat(colors, 1)

        output = {'color': color}
        if depth_mode is not None:
            output['depth'] = self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cpu
            )
        return output

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
        cpu: bool = False,
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        depths = []
        for start in range(0, v, self.cfg.view_batch):
            end = min(start + self.cfg.view_batch, v)
            vb = end - start
            depth = render_depth_cuda(
                rearrange(extrinsics[:, start:end], "b vb i j -> (b vb) i j"),
                rearrange(intrinsics[:, start:end], "b vb i j -> (b vb) i j"),
                rearrange(near[:, start:end], "b vb -> (b vb)"),
                rearrange(far[:, start:end], "b vb -> (b vb)"),
                image_shape,
                repeat(gaussians.means, "b g xyz -> (b vb) g xyz", vb=vb),
                repeat(gaussians.covariances, "b g i j -> (b vb) g i j", vb=vb),
                repeat(gaussians.opacities, "b g -> (b vb) g", vb=vb),
                mode=mode,
            )
            depth = rearrange(depth, "(b vb) h w -> b vb h w", b=b, vb=vb)
            if cpu:
                depth = depth.cpu()
            depths.append(depth)
        result = torch.cat(depths, 1)
        return result

    def render_pano(
        self,
        gaussians: Gaussians | List[Gaussians] | List[List[Gaussians]],
        extrinsics: Float[Tensor, "batch view 4 4"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        depth_mode: DepthRenderingMode | None = None,
        face_idx: int | None = None,
        cpu: bool = False,
        context_extrinsics: Float[Tensor, "batch cview 4 4"] | None = None,
    ):
        # Render multiple panos
        if isinstance(gaussians, list):
            outputs = []
            for g in gaussians:
                outputs.append(self.render_pano(
                    g, extrinsics, near, far, depth_mode, face_idx, cpu, context_extrinsics
                ))

            if not isinstance(g, list) and context_extrinsics is not None:
                outputs_new = {}
                if 'color' in outputs[0]:
                    target = repeat(extrinsics[:, :, :3, 3], "... v d -> ... v vc d", vc=context_extrinsics.shape[1])
                    context = repeat(context_extrinsics[:, :, :3, 3], "... vc d -> ... v vc d", v=extrinsics.shape[1])
                    dist = torch.norm(target - context, dim=-1)
                    total = dist.sum(-1, keepdim=True)
                    weights = 1 - dist / total
                    outputs_new['color'] = sum(
                        output['color'] * weight[..., None, None, None].to(output['color'].device)
                        for output, weight in zip(outputs, weights.unbind(-1))
                    )
                if 'depth' in outputs[0]:
                    outputs_new['depth'] = torch.stack([output['depth'] for output in outputs], -1).max(-1).values
                return outputs_new

            return outputs

        # Prepare cubemap extrinsics and intrinsics
        b, v = extrinsics.shape[:2]
        if face_idx is None:
            num_faces = self.num_faces
            cubemap_Rs = self.cubemap_Rs
        else:
            num_faces = 1
            cubemap_Rs = self.cubemap_Rs[face_idx:face_idx + 1]
        cam2world = repeat(extrinsics, "b v i j -> b (v f) i j", f=num_faces)
        cubemap_Rs = repeat(cubemap_Rs, "f i j -> b (v f) i j", b=b, v=v)
        extrinsics = cam2world @ cubemap_Rs
        intrinsics = repeat(self.intrinsics, "i j -> b (v f) i j", b=b, v=v, f=num_faces)
        near = repeat(near, "b v -> b (v f)", f=num_faces)
        far = repeat(far, "b v -> b (v f)", f=num_faces)

        # Render cubemap faces
        output = self.forward(
            gaussians, extrinsics, intrinsics, near, far, (self.face_w, self.face_w), depth_mode, cpu
        )

        # Stich cubemap faces
        for key in ("color", "depth"):
            if key not in output:
                continue
            if face_idx is None:
                cubemap = rearrange(output[key], "b (v f) ... h w -> (b v) f ... h w", f=self.num_faces)
            else:
                # Warp faces to equirectangular
                faces = rearrange(output['color'], "b v ... h w -> (b v) 1 ... h w")
                cubemap = [torch.zeros_like(faces)] * self.num_faces
                cubemap[face_idx] = faces
                cubemap = torch.cat(cubemap, 1)
            # from PIL import Image
            # Image.fromarray((cubemap[0][0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(f"debug/{key}.png")

            if key == 'depth':
                cubemap = cubemap * self.depth_norm.to(cubemap.device)
                cubemap = rearrange(cubemap, "bv f h w -> bv f 1 h w")

            pano = sample_cubefaces(cubemap, self.tp, self.coor_y, self.coor_x)
            pano = rearrange(pano, "(b v) ... -> b v ...", b=b)
            # from PIL import Image
            # Image.fromarray((pano[0][0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save("debug/pano.png")

            if key == 'depth':
                pano = rearrange(pano, "b v 1 h w -> b v h w")

            output[key] = pano

        if face_idx is not None:
            mask = self.tp == face_idx
            # from PIL import Image
            # Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(f"debug/mask_{face_idx}.png")
            output['mask'] = mask

        return output


def uv_grid(h, w, device):
    u, v = torch.meshgrid(
        torch.linspace(0.5 / w, 1 - 0.5 / w, w, device=device),
        torch.linspace(0.5 / h, 1 - 0.5 / h, h, device=device),
        indexing='xy'
    )
    return u, v


def erp_uv_grid(h, w, device):
    u, v = uv_grid(h, w, device)
    phi = u * 2 * np.pi - np.pi
    theta = -v * np.pi - np.pi / 2
    return phi, theta


def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)


def cubmap_mapping(face_w, h, w):
    u, v = erp_uv_grid(h, w, 'cpu')
    u = u.numpy()
    v = v.numpy()

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    return tp, coor_x, coor_y


def sample_cubefaces(
    cube_faces: Float[Tensor, "batch 6 channel face_h face_w"],
    tp: Float[Tensor, "height width"],
    coor_y: Float[Tensor, "height width"],
    coor_x: Float[Tensor, "height width"],
):
    pad_ud = torch.stack([
        torch.stack([cube_faces[:, 5, :, 0, :], cube_faces[:, 4, :, -1, :]], -2),
        torch.stack([cube_faces[:, 5, :, :, -1], cube_faces[:, 4, :, :, -1].flip(-1)], -2),
        torch.stack([cube_faces[:, 5, :, -1, :].flip(-1), cube_faces[:, 4, :, 0, :].flip(-1)], -2),
        torch.stack([cube_faces[:, 5, :, :, 0].flip(-1), cube_faces[:, 4, :, :, 0]], -2),
        torch.stack([cube_faces[:, 0, :, 0, :], cube_faces[:, 2, :, 0, :].flip(-1)], -2),
        torch.stack([cube_faces[:, 2, :, -1, :].flip(-1), cube_faces[:, 0, :, -1, :]], -2),
    ], 1)
    cube_faces = torch.cat([cube_faces, pad_ud], -2)

    pad_lr = torch.stack([
        torch.stack([cube_faces[:, 1, :, :, 0], cube_faces[:, 3, :, :, -1]], -1),
        torch.stack([cube_faces[:, 2, :, :, 0], cube_faces[:, 0, :, :, -1]], -1),
        torch.stack([cube_faces[:, 3, :, :, 0], cube_faces[:, 1, :, :, -1]], -1),
        torch.stack([cube_faces[:, 0, :, :, 0], cube_faces[:, 2, :, :, -1]], -1),
        F.pad(
            torch.stack([cube_faces[:, 1, :, 0, :].flip(-1), cube_faces[:, 3, :, 0, :]], -1),
            (0, 0, 1, 1),
            'constant',
            0
        ),
        F.pad(
            torch.stack([cube_faces[:, 1, :, -2, :], cube_faces[:, 3, :, -2, :].flip(-1)], -1),
            (0, 0, 1, 1),
            'constant',
            0
        )
    ], 1)
    cube_faces = torch.cat([cube_faces, pad_lr], -1)
    cube_faces = torch.roll(cube_faces, 1, -1)
    cube_faces = torch.roll(cube_faces, 1, -2)

    b, face, _, face_h, face_w = cube_faces.shape
    cube_faces = rearrange(cube_faces, "b face c h w -> b c (face h) w")
    coor_x = (coor_x + 1) / face_w * 2 - 1
    coor_y = ((coor_y + 1 + tp * face_h) / face_h / face) * 2 - 1
    grid = torch.stack([coor_x, coor_y], -1)
    grid = grid.to(cube_faces.device)
    grid = repeat(grid, "h w i -> b h w i", b=b)
    pano = F.grid_sample(cube_faces, grid, mode='bilinear', align_corners=False)
    return pano

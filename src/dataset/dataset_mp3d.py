from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler

from src.model.encoder.unifuse.datasets.util import Equirec2Cube
from einops import rearrange
import torch.nn.functional as F


@dataclass
class DatasetMP3DCfg(DatasetCfgCommon):
    name: Literal["mp3d"]
    roots: list[Path]
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_datasets: list[dict]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = False


class DatasetMP3D(Dataset):
    cfg: DatasetMP3DCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetMP3DCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # scan folders in cfg.roots[0]
        if stage == "predict":
            stage = "test"

        height = cfg.image_shape[0]
        height = max(height, 512)
        resolution = (height * 2, height)
        resolution = 'x'.join(map(str, resolution))
        if stage == "test":
            self.roots = []
            for test_dataset in cfg.test_datasets:
                name = test_dataset["name"]
                dis = test_dataset["dis"]
                self.roots.append(
                    cfg.roots[0] / f"png_render_{stage}_{resolution}_seq_len_3_{name}_dist_{dis}"
                )
        else:
            self.roots = [r / f"png_render_{stage}_{resolution}_seq_len_3_m3d_dist_0.5" for r in cfg.roots]

        data = []
        for root, test_dataset in zip(self.roots, cfg.test_datasets):
            if not os.path.exists(root):
                continue
            scenes = os.listdir(root)
            scenes.sort()
            for s in scenes:
                data.append({
                    'root': root,
                    'scene_id': s,
                    'name': test_dataset["name"],
                    'dis': test_dataset["dis"],
                    'baseline': test_dataset["dis"] * 2,
                })
        self.data = data

        if self.cfg.overfit_to_scene is not None:
            data = [d for d in self.data if d["scene_id"] == self.cfg.overfit_to_scene]
            self.data = data * (len(self.data) // 100)

        self.e2c_mono = Equirec2Cube(512, 1024, 256)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        scene = data['scene_id']
        scene_path = data['root'] / scene
        views = os.listdir(scene_path)
        views.sort()

        # Load the images.
        rgbs_path = [str(scene_path / v / 'rgb.png') for v in views]
        context_indices = torch.tensor([0, 2])
        target_indices = torch.tensor([1])
        context_images = [rgbs_path[i] for i in context_indices]
        target_images = [rgbs_path[i] for i in target_indices]
        context_images = self.convert_images(context_images)
        target_images = self.convert_images(target_images)

        # Load the depth.
        if data['name'] == 'm3d':
            depths_path = [str(scene_path / v / 'depth.png') for v in views]
            context_depths = [depths_path[i] for i in context_indices]
            target_depths = [depths_path[i] for i in target_indices]
            context_depths = self.convert_images(context_depths)
            target_depths = self.convert_images(target_depths)
            context_depths = context_depths.float() / 1000.0
            target_depths = target_depths.float() / 1000.0
            context_depths = context_depths.clamp(min=0.)
            target_depths = target_depths.clamp(min=0.)
            context_mask = (context_depths > self.near) & (context_depths < self.far)
            target_mask = (target_depths > self.near) & (target_depths < self.far)

        # load camera
        trans_path = [scene_path / v / 'tran.txt' for v in views]
        rots_path = [scene_path / v / 'rot.txt' for v in views]
        trans = []
        rots = []
        for tran_path, rot_path in zip(trans_path, rots_path):
            trans.append(np.loadtxt(tran_path))
            rots.append(np.loadtxt(rot_path))
        trans = torch.tensor(trans)
        rots = torch.tensor(rots)
        extrinsics = self.convert_poses(trans, rots)

        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
            a, b = context_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        intrinsics = torch.eye(3, dtype=torch.float32)
        fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics = repeat(intrinsics, "h w -> b h w", b=len(extrinsics)).clone()

        # resize images for mono depth
        mono_images = F.interpolate(context_images, size=(256, 512), mode='bilinear')
        mono_images = F.interpolate(mono_images, size=(512, 1024), mode='bilinear')

        # Project the images to the cube.
        cube_image = []
        for img in mono_images:
            img = img.numpy()
            img = rearrange(img, "c h w -> h w c")
            img = self.e2c_mono.run(img)
            cube_image.append(img)
        cube_image = np.stack(cube_image)
        cube_image = rearrange(cube_image, "v h w c -> v c h w")

        nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
        data.pop('root')
        data.update({
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "mono_image": mono_images,
                "cube_image": cube_image,
                "near": self.get_bound("near", len(context_images)) / nf_scale,
                "far": self.get_bound("far", len(context_images)) / nf_scale,
                "index": context_indices,
                # "depth": context_depths,
                # "mask": context_mask,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
                # "depth": target_depths,
                # "mask": target_mask,
            },
            "scene": scene,
        })
        if data['name'] == 'm3d':
            data["context"]["depth"] = context_depths
            data["context"]["mask"] = context_mask
            data["target"]["depth"] = target_depths
            data["target"]["mask"] = target_mask

        return data

    def convert_poses(
        self,
        trans: Float[Tensor, "batch 3"],
        rots: Float[Tensor, "batch 3 3"],
    ) -> Float[Tensor, "batch 4 4"]:  # extrinsics
        b, _ = trans.shape

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        c2w = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        c2w[:, :3, :3] = rots
        c2w[:, :3, 3] = trans
        w2w = torch.tensor([  # X -> X, -Z -> Y, upY -> Z
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]).float()
        c2c = torch.tensor([  # rightx -> rightx, upy -> -downy, backz -> -forwardz
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).float()
        c2w = w2w @ c2w @ c2c
        return c2w

    def convert_images(
        self,
        images: list[str],
    ):
        torch_images = []
        for image in images:
            image = Image.open(image)
            image = image.resize(self.cfg.image_shape[::-1], Image.LANCZOS)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.data)

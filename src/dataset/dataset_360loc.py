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
from torch.utils.data import IterableDataset
import numpy as np

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler

from src.model.encoder.unifuse.datasets.util import Equirec2Cube
from einops import rearrange
import torch.nn.functional as F
import json
from functools import cached_property


@dataclass
class Dataset360LocCfg(DatasetCfgCommon):
    name: Literal["360loc"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_chunk_interval: int
    train_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = False
    cache_images: bool = True


class Dataset360Loc(IterableDataset):
    cfg: Dataset360LocCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    data: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: Dataset360LocCfg,
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

        if stage == "train":
            locations = ['concourse', 'hall', 'piatrium']
        else:
            locations = ['atrium']
        assert len(cfg.roots) == 1
        root = cfg.roots[0]
        self.data = []
        for location in locations:
            seqs = [list((root / location / folder).glob('*360*/')) for folder in ('mapping', 'query_360')]
            seqs = sum(seqs, [])
            self.data.extend(seqs)

        if self.cfg.overfit_to_scene is not None:
            self.data = [root / self.cfg.overfit_to_scene]

        self.e2c_mono = Equirec2Cube(512, 1024, 256)

        self.times_per_scene = self.cfg.train_times_per_scene if self.stage == "train" \
            else self.view_sampler.cfg.test_times_per_scene

        self.load_images = True

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def load_extrinsics(self, example_path):
        example = example_path / 'camera_pose.json'
        with open(example) as f:
            example = json.load(f)
        frames, extrinsics_orig = list(example.keys()), list(example.values())
        extrinsics_orig = torch.tensor(extrinsics_orig)
        return frames, extrinsics_orig

    @cached_property
    def total_frames(self):
        extrinsics = [self.load_extrinsics(example)[1] for example in self.data]
        return sum(len(e) for e in extrinsics)

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.data = self.shuffle(self.data)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage != "train" and worker_info is not None:
            self.data = [
                example
                for data_idx, example in enumerate(self.data)
                if data_idx % worker_info.num_workers == worker_info.id
            ]

        for example_path in self.data:
            frames, extrinsics_orig = self.load_extrinsics(example_path)
            scene = f"{example_path.parts[-3]}-{example_path.parts[-1]}"

            if self.cfg.cache_images and self.stage == "train" and self.load_images:
                images = [example_path / 'image' / frame for frame in frames]
                images = self.convert_images(images)

            for i in range(self.times_per_scene):
                context_indices, target_indices = self.view_sampler.sample(
                    scene,
                    extrinsics_orig,
                    i=i,
                )
                if context_indices is None:
                    break

                load_target = (target_indices >= 0).all()

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics_orig[context_indices]
                if load_target:
                    target_extrinsics = extrinsics_orig[target_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    context_extrinsics[:, :3, 3] /= scale
                    if load_target:
                        target_extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                intrinsics = torch.eye(3, dtype=torch.float32)
                fx, fy, cx, cy = 0.25, 0.5, 0.5, 0.5
                intrinsics[0, 0] = fx
                intrinsics[1, 1] = fy
                intrinsics[0, 2] = cx
                intrinsics[1, 2] = cy

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                data = {
                    "context": {
                        "extrinsics": context_extrinsics,
                        "intrinsics": repeat(intrinsics, "h w -> b h w", b=len(context_indices)),
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "scene": scene,
                }

                if load_target:
                    data["target"] = {
                        "extrinsics": target_extrinsics,
                        "intrinsics": repeat(intrinsics, "h w -> b h w", b=len(target_indices)),
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    }

                # Load the images.
                if self.load_images:
                    if self.cfg.cache_images and self.stage == "train":
                        context_images = images[context_indices]
                        if load_target:
                            target_images = images[target_indices]
                    else:
                        context_images = [example_path / 'image' / frames[i] for i in context_indices]
                        context_images = self.convert_images(context_images)
                        if load_target:
                            target_images = [example_path / 'image' / frames[i] for i in target_indices]
                            target_images = self.convert_images(target_images)

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

                    data["context"]["image"] = context_images
                    data["context"]["mono_image"] = mono_images
                    data["context"]["cube_image"] = cube_image
                    if load_target:
                        data["target"]["image"] = target_images

                yield data

    def convert_images(
        self,
        images: list[str | Path],
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
        return len(self.data) * self.times_per_scene


from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerSequenceCfg:
    name: Literal["sequence"]
    start_frame: int
    skip_frame: int
    test_times_per_scene: int
    overlap_frame: int


class ViewSamplerSequence(ViewSampler[ViewSamplerSequenceCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        device: torch.device = torch.device("cpu"),
        i: int = 0,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        index_context_left = self.cfg.start_frame + i * (self.cfg.skip_frame - self.cfg.overlap_frame)
        index_context_right = index_context_left + self.cfg.skip_frame
        index_target = -1

        return (
            torch.tensor((index_context_left, index_context_right)),
            torch.tensor([index_target]),
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return 1

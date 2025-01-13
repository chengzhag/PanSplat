from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    test_times_per_scene: int
    chosen: dict | None = None


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        device: torch.device = torch.device("cpu"),
        i: int = 0,
    ) -> tuple[
        Int64[Tensor, " context_view"] | None,  # indices for context views
        Int64[Tensor, " target_view"] | None,  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            min_gap = max_gap = (self.cfg.max_distance_between_context_views
                                 + self.cfg.min_distance_between_context_views) // 2
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        # NOTE: we keep the bug untouched to follow initial pixelsplat cfgs
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, min_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        if self.cfg.chosen is not None:
            if len(self.cfg.chosen[scene]) > i:
                index_context_left = self.cfg.chosen[scene][i]
            else:
                return None, None
        elif self.stage == "test":
            index_context_left = (num_views - context_gap - 1) * i / max((self.cfg.test_times_per_scene - 1), 1)
            index_context_left = int(index_context_left)
        else:
            index_context_left = torch.randint(
                num_views if self.cameras_are_circular else num_views - context_gap,
                size=tuple(),
                device=device,
            ).item()
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )

            # # When testing, pick the middle view.
            # index_target = torch.tensor(
            #     [(index_context_left + index_context_right) // 2],
            #     device=device,
            # )

            # # When testing, rotate from left to right.
            # index_target = torch.tensor(
            #     [index_context_left + i % (index_context_right - index_context_left + 1)],
            #     device=device,
            # )
        else:
            # When training or validating (visualizing), pick at random.
            index_target = torch.randint(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right + 1 - self.cfg.min_distance_to_context_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        return (
            torch.tensor((index_context_left, index_context_right)),
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views

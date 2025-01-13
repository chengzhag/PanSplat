from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: dict | None,
        batch: BatchedExample,
        encoder_outputs: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        delta = prediction['color'] - image
        loss = (delta**2).mean()

        return self.cfg.weight * loss

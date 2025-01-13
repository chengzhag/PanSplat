from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from .loss import Loss

from einops import rearrange


@dataclass
class LossMVDepthCfg:
    weight: float
    gamma: float


@dataclass
class LossMVDepthCfgWrapper:
    mvdepth: LossMVDepthCfg


class LossMVDepth(Loss[LossMVDepthCfg, LossMVDepthCfgWrapper]):
    def forward(
        self,
        prediction: dict | None,
        batch: BatchedExample,
        encoder_outputs: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        mvs_outputs = encoder_outputs['mvs_outputs']
        depth_gt = batch['context']['depth']
        mask_gt = batch['context']['mask']
        depth_loss = 0.0
        if 'stages' in mvs_outputs:
            n_predictions = len([s for s in mvs_outputs['stages'] if 'depth' in s])
            for i, stage in enumerate(mvs_outputs['stages']):
                if 'depth' not in stage:
                    continue
                i_weight = self.cfg.gamma ** (n_predictions - i - 1)
                depth_pred = stage['depth']
                depth_pred = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode='bilinear', align_corners=False)
                depth_pred = rearrange(depth_pred, 'b v ... -> b v 1 ...')

                i_loss = (1. / depth_pred[mask_gt] - 1. / depth_gt[mask_gt]).abs().mean()
                depth_loss += i_weight * i_loss
        else:
            depth_pred = mvs_outputs['depth']
            depth_pred = rearrange(depth_pred, 'b v ... -> b v 1 ...')
            depth_loss = (1. / depth_pred[mask_gt] - 1. / depth_gt[mask_gt]).abs().mean()

        return self.cfg.weight * depth_loss

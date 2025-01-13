from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfg, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfg, LossMseCfgWrapper
from src.model.decoder import DecoderCfg
from src.dataset.data_module import DatasetCfg
from src.model.encoder.gaussian_head import GaussianHead
from jaxtyping import install_import_hook
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.model.decoder import get_decoder
from src.global_cfg import get_cfg


@dataclass
class LossPyimageCfg:
    gaussian_head: dict
    decoder: DecoderCfg
    lpips: LossLpipsCfg
    mse: LossMseCfg
    gamma: float


@dataclass
class LossPyimageCfgWrapper:
    pyimage: LossPyimageCfg


class LossPyimage(Loss[LossPyimageCfg, LossPyimageCfgWrapper]):
    def __init__(self, cfg: LossPyimageCfgWrapper, dataset_cfg: DatasetCfg) -> None:
        super().__init__(cfg, dataset_cfg)
        self.gaussian_head = GaussianHead(
            **self.cfg.gaussian_head,
            image_height=get_cfg().dataset.image_shape[0],
            encoder_cfg=get_cfg().model.encoder,
        )

        self.decoder = get_decoder(self.cfg.decoder, self.dataset_cfg)

        self.lpips = LossLpips(LossLpipsCfgWrapper(self.cfg.lpips), dataset_cfg)
        self.mse = LossMse(LossMseCfgWrapper(self.cfg.mse), dataset_cfg)

    def forward(
        self,
        prediction: dict | None,
        batch: BatchedExample,
        encoder_outputs: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        outputs = self.render(batch, encoder_outputs)
        n_predictions = len(outputs)
        loss = 0.0

        for i, output in enumerate(outputs):
            i_weight = self.cfg.gamma ** (n_predictions - i - 1)
            i_loss = 0.0
            for loss_fn in (self.lpips, self.mse):
                i_loss += loss_fn.forward(output, batch, encoder_outputs, global_step)
            loss += i_weight * i_loss

        return loss

    def render(
        self,
        batch: BatchedExample,
        encoder_outputs: dict,
    ):
        gaussians = self.gaussian_head(encoder_outputs["mvs_outputs"], batch["context"])
        gaussians = [s['gaussians'] for s in gaussians['stages']]

        output_stages = self.decoder.render_pano(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            context_extrinsics=batch["context"]["extrinsics"],
        )

        return output_stages

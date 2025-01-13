from typing import Optional

from .encoder import Encoder
from .encoder_pansplat import EncoderPanSplat, EncoderPanSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_pansplat import EncoderVisualizerPanSplat

ENCODERS = {
    "pansplat_enc": (EncoderPanSplat, EncoderVisualizerPanSplat),
}

EncoderCfg = EncoderPanSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer

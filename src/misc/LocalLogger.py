import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
import moviepy.editor as mpy


class LocalLogger(Logger):
    def __init__(self, save_dir: Path) -> None:
        super().__init__()
        self.experiment = None
        self._save_dir = save_dir
        self._log_dir = save_dir / "local"

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_dir / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)

    @rank_zero_only
    def log_video(
        self,
        key: str,
        video: Any,
        step: Optional[int] = None,
    ):
        tensor = video._prepare_video(video.data)
        clip = mpy.ImageSequenceClip(list(tensor), fps=video._fps)
        video_dir = self.log_dir / key
        video_dir.mkdir(exist_ok=True, parents=True)
        clip.write_videofile(
            str(video_dir / f"{step:0>6}.mp4"), logger=None
        )

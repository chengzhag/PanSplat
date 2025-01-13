from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_mp3d import DatasetMP3D, DatasetMP3DCfg
from .dataset_360loc import Dataset360Loc, Dataset360LocCfg
from .dataset_insta360 import DatasetInsta360, DatasetInsta360Cfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "mp3d": DatasetMP3D,
    "360loc": Dataset360Loc,
    "insta360": DatasetInsta360,
}


DatasetCfg = DatasetMP3DCfg | Dataset360LocCfg | DatasetInsta360Cfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)

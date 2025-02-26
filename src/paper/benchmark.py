import os
from pathlib import Path
import warnings
from datetime import timedelta

import hydra
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import default_collate
from einops import repeat, rearrange
from tqdm.auto import tqdm
from collections import defaultdict
from ..misc.benchmarker import Benchmarker
import numpy as np
import json
from pytorch_lightning import Trainer
import tempfile
from pytorch_lightning.callbacks import Callback


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.dataset import get_dataset
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg_dict.model.weights_path = None
    cfg_dict.dataset.cache_images = False
    cfg_dict.data_loader.train.batch_size = 1
    cfg_dict.data_loader.train.num_workers = 0
    torch.manual_seed(cfg_dict.seed)

    # Sweep over resolutions.
    height_max = 2048
    height_min = 256
    heights_special = np.array([256, 512, 1024, 2048])
    # resolution_percentage_min = (height_min / height_max) ** 2 * 100
    # resolution_percentages = np.linspace(resolution_percentage_min, 100., 6)
    # height_ratio = (resolution_percentages / 100.) ** 0.5
    # heights = (height_ratio * height_max).astype(int)
    heights = np.linspace(height_min, height_max, 8)
    heights = np.unique(np.concatenate([heights_special, heights]))
    heights = heights // 32 * 32
    widths = 2 * heights
    resolution_percentages = (heights / height_max) ** 2 * 100

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    for height, width, resolution_percentage in zip(heights, widths, resolution_percentages):
        height, width = int(height), int(width)
        key = f"{height}x{width}"
        cfg_dict.dataset.image_shape = [height, width]
        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        resolution_dir = output_dir / key

        print("-" * 80)
        print(f"Resolution: {key}, {resolution_percentage:.2f}%, training...")
        print("-" * 80)
        try:
            benchmark_train(cfg, resolution_dir)
        except Exception as e:
            print(f"Error during training: {e}")
        torch.cuda.empty_cache()

        print("-" * 80)
        print(f"Resolution: {key}, {resolution_percentage:.2f}%, inference...")
        print("-" * 80)
        try:
            with torch.inference_mode():
                benchmark_inference(cfg, resolution_dir)
        except Exception as e:
            print(f"Error during inference: {e}")
        torch.cuda.empty_cache()

        resolution_dir.mkdir(parents=True, exist_ok=True)
        with open(resolution_dir / "info.json", "w") as f:
            json.dump({
                "key": key,
                "height": height,
                "width": width,
                "percentage": resolution_percentage,
                "special": height in heights_special,
            }, f, indent=4)


def benchmark_train(cfg, output_dir):
    num_batches = 100
    step_tracker = StepTracker()
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        cfg.val,
        cfg.predict,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss, cfg.dataset),
        step_tracker,
        cfg.model.weights_path,
        cfg.model.wo_defbp2,
    )

    class ResetMemoryStats(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            benchmarker.reset_memory_stats()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            benchmarker.log_memory_stats()

    with tempfile.TemporaryDirectory() as temp_dir:
        logger = LocalLogger(
            save_dir=Path(temp_dir),
        )

        trainer = Trainer(
            accelerator="gpu",
            logger=logger,
            devices="auto",
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
            callbacks=[ResetMemoryStats()],
            val_check_interval=None,
            check_val_every_n_epoch=None,
            enable_progress_bar=True,
            gradient_clip_val=cfg.trainer.gradient_clip_val if model_wrapper.automatic_optimization else None,
            max_epochs=None,
            max_steps=num_batches,
            num_sanity_val_steps=0,
            limit_train_batches=num_batches,
            limit_val_batches=0,
            limit_test_batches=0,
            precision=cfg.trainer.precision,
        )

        data_module = DataModule(
            cfg.dataset,
            cfg.data_loader,
            step_tracker,
            global_rank=trainer.global_rank,
        )

        benchmarker = Benchmarker()
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
        )

        benchmarker.dump(output_dir / "train.json")
        benchmarker.summarize()


def benchmark_inference(cfg, output_dir):
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        cfg.val,
        cfg.predict,
        encoder,
        encoder_visualizer,
        decoder,
        [],
        None,
        cfg.model.weights_path,
        cfg.model.wo_defbp2,
    )
    model_wrapper.eval()
    model_wrapper = model_wrapper.cuda()

    dataset = get_dataset(cfg.dataset, "test", None)
    device = model_wrapper.device
    benchmarker = Benchmarker()
    num_renders = 1
    num_batches = 100

    # render frames
    i = 0
    for batch in tqdm(dataset, desc="Benchmarking", total=num_batches, leave=False):
        if num_batches and i >= num_batches:
            break
        i += 1
        benchmarker.reset_memory_stats()

        batch = default_collate([batch])
        batch = apply_to_collection(batch, Tensor, lambda x: x.to(device))

        with benchmarker.time("encoder"):
            gaussians_prob = model_wrapper.encoder(batch["context"], inference=True)["gaussians"]["gaussians"]

        near = repeat(batch["target"]["near"][:, 0], "b -> b v", v=num_renders)
        far = repeat(batch["target"]["far"][:, 0], "b -> b v", v=num_renders)
        extrinsics = repeat(batch["target"]["extrinsics"][:, 0], "b ... -> b v ...", v=num_renders)
        context_extrinsics = batch["context"]["extrinsics"]
        del batch
        with benchmarker.time("decoder", num_renders):
            model_wrapper.decoder.render_pano(
                gaussians_prob,
                extrinsics,
                near,
                far,
                depth_mode=None,
                # cpu=True,
                context_extrinsics=context_extrinsics,
            )

        del gaussians_prob, extrinsics, near, far
        # torch.cuda.empty_cache()
        benchmarker.log_memory_stats()

    benchmarker.dump(output_dir / "inference.json")
    benchmarker.summarize()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]

    main()

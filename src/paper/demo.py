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
from einops import pack, repeat, rearrange
from tqdm.auto import tqdm
import yaml
from collections import defaultdict
import cv2
import tempfile
import subprocess
import moviepy.editor as mpy
from PIL import Image


from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_sequence import ViewSamplerSequenceCfg
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

from src.visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def demo(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)

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

    # This allows the current step to be shared with the data loader processes.
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

    # load segmentation configuration
    name = cfg.dataset.name
    seg_cfg = f"src/paper/{name}.yaml"
    with open(seg_cfg, "r") as f:
        args = yaml.safe_load(f)
    fps = 24

    if cfg.dataset.name == "mp3d":
        cfg.dataset.view_sampler.test_times_per_scene = args.get('num_videos_per_scene', None)
        if 'test_datasets' in args:
            cfg.dataset.test_datasets = args['test_datasets']
        predict(args, model_wrapper, cfg, output_dir, fps)
    else:
        raise NotImplementedError


def predict(args, model_wrapper, cfg, output_dir, fps):
    device = model_wrapper.device
    dataset = get_dataset(cfg.dataset, "test", None)
    for batch in tqdm(
        dataset,
        desc="Predicting",
    ):
        batch = default_collate([batch])
        if "depth" in batch["context"]:
            del batch["context"]["depth"], batch["context"]["mask"]
        batch["context"] = apply_to_collection(batch["context"], Tensor, lambda x: x.to(device))
        gaussians_prob = model_wrapper.encoder(batch["context"], inference=True)["gaussians"]["gaussians"]

        near = batch["context"]["near"][:, 0]
        far = batch["context"]["far"][:, 0]
        context_extrinsics = batch["context"]["extrinsics"]
        scene_id = batch["scene"][0]
        context_indices = batch["context"]["index"][0]
        frame_str = "_".join([str(x.item()) for x in context_indices])
        context_images = batch["context"]["image"][0].cpu()
        del batch

        # save context images
        for context_image, idx in zip(context_images, context_indices):
            context_image = (context_image.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
            context_image = rearrange(context_image, "c h w -> h w c")
            context_image = Image.fromarray(context_image)
            output_file = output_dir / f"{scene_id}-{frame_str}/context_{idx}.png"
            output_file.parent.mkdir(exist_ok=True, parents=True)
            context_image.save(output_file)

        # save middle frame
        t = torch.tensor([0.5], device=device)
        extrinsics = interpolate_extrinsics(
            context_extrinsics[:, 0],
            context_extrinsics[:, 1],
            t,
        )
        n = repeat(near, "b -> b v", v=1)
        f = repeat(far, "b -> b v", v=1)
        output_prob = model_wrapper.decoder.render_pano(
            gaussians_prob, extrinsics, n, f, None, context_extrinsics=context_extrinsics
        )
        output_frame = output_prob['color'][0, 0]
        output_file = output_dir / f"{scene_id}-{frame_str}/middle.png"
        output_frame = (output_frame.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        output_frame = rearrange(output_frame, "c h w -> h w c")
        output_frame = Image.fromarray(output_frame)
        output_frame.save(output_file)

        def render_video_generic(trajectory_fn, num_frames, output_file):
            t = torch.linspace(0, 1, num_frames, device=device)
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
            extrinsics = trajectory_fn(t)
            n = repeat(near, "b -> b v", v=num_frames)
            f = repeat(far, "b -> b v", v=num_frames)

            # render frames
            output_prob = model_wrapper.decoder.render_pano(
                gaussians_prob, extrinsics, n, f, None, cpu=True, context_extrinsics=context_extrinsics
            )
            output_frames = output_prob['color'][0]

            # save nearest gt frame
            gt_indices = torch.stack([t, 1 - t]).argmin(dim=0).cpu()
            nearest_frames = context_images[gt_indices]
            nearest_dir = output_file.parent / f"{output_file.stem}-nearest"

            for video, video_dir in ((output_frames, output_file), (nearest_frames, nearest_dir)):
                video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
                video = pack([video, video[::-1][1:-1]], "* c h w")[0]
                video = rearrange(video, "n c h w -> n h w c")
                clip = mpy.ImageSequenceClip(list(video), fps=fps)
                video_dir.parent.mkdir(exist_ok=True, parents=True)
                video_dir = video_dir.with_suffix('.mp4')
                clip.write_videofile(str(video_dir), logger=None)

        def wobble_trajectory_fn(t):
            origin_a = context_extrinsics[:, 0, :3, 3]
            origin_b = context_extrinsics[:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                context_extrinsics[:, 0],
                delta * args["wobble_radius"],
                t,
            )
            return extrinsics

        render_video_generic(
            wobble_trajectory_fn,
            args["num_render_frames"],
            output_dir / f"{scene_id}-{frame_str}/wobble"
        )

        def interpolate_trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                context_extrinsics[:, 0],
                context_extrinsics[:, 1],
                t,
            )
            return extrinsics

        render_video_generic(
            interpolate_trajectory_fn,
            args["num_render_frames"],
            output_dir / f"{scene_id}-{frame_str}/interpolate"
        )

        del gaussians_prob, near, far, context_extrinsics
        torch.cuda.empty_cache()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]

    with torch.inference_mode():
        demo()
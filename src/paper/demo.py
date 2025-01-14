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
    if cfg.mode == "predict":
        name += f"_{cfg.mode}"
    seg_cfg = f"src/paper/{name}.yaml"
    with open(seg_cfg, "r") as f:
        args = yaml.safe_load(f)
    fps = 24

    if cfg.mode == "all":
        dataset = get_dataset(cfg.dataset, "test", None)
        arg = args
        args = []
        for scene in dataset.data:
            arg["scene"] = scene.relative_to(cfg.dataset.roots[0])
            args.append(arg.copy())
    elif cfg.mode == "predict":
        cfg.dataset.view_sampler.test_times_per_scene = args.get('num_videos_per_scene', None)
        if 'scenes' in args:
            scenes = {k: [int(s.split('_')[0]) for s in v] for k, v in args['scenes'].items()}
            cfg.dataset.view_sampler.chosen = scenes
        if 'test_datasets' in args:
            cfg.dataset.test_datasets = args['test_datasets']
        predict(args, model_wrapper, cfg, output_dir, fps)
        return

    for arg in tqdm(args, desc="Rendering videos"):
        render_video(arg, model_wrapper, cfg, output_dir, fps)


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


def render_video(args, model_wrapper, cfg, output_dir, fps):
    device = model_wrapper.device
    cfg.dataset.overfit_to_scene = args["scene"]
    all_frames = args["end_frame"] is None
    if all_frames:
        dataset = get_dataset(cfg.dataset, "test", None)
        args["end_frame"] = dataset.total_frames - 1
    num_batchs = (
        args["end_frame"] - args["start_frame"] - args["overlap_frame"]
    ) // (args["skip_frame"] - args["overlap_frame"])
    if all_frames:
        args["num_render_frames"] = args["num_render_frames"] * num_batchs

    # load dataset
    view_sampler_cfg = ViewSamplerSequenceCfg(
        "sequence",
        args["start_frame"],
        args["skip_frame"],
        num_batchs,
        args["overlap_frame"],
    )
    cfg.dataset.view_sampler = view_sampler_cfg
    dataset = get_dataset(cfg.dataset, "test", None)
    dataset.load_images = False

    # load context information
    context_extrinsics = []
    context_indices = []
    for batch in tqdm(
        dataset,
        desc="Loading data",
        leave=False,
    ):
        context_extrinsics.append(batch["context"]["extrinsics"])
        context_indices.append(batch["context"]["index"])
        scene_id = batch["scene"]
    context_extrinsics = torch.stack(context_extrinsics)
    context_indices = torch.stack(context_indices)
    end_frame = context_indices.max()
    context_indices = context_indices - context_indices.min()
    context_locations = context_extrinsics[..., :3, 3]

    # landmarks to associate target frames with context
    _, landmark_orders = torch.unique(context_indices, return_inverse=True)
    landmark_odometers = torch.zeros_like(landmark_orders, dtype=torch.float32)
    odometer = 0.
    last_landmark_location = context_locations[0, 0]
    for i in range(1, landmark_orders.max() + 1):
        landmark_location = context_locations[landmark_orders == i][0]
        odometer += (landmark_location - last_landmark_location).norm()
        landmark_odometers[landmark_orders == i] = odometer
        last_landmark_location = landmark_location

    # interpolate time
    t = torch.linspace(0, 1, args["num_render_frames"], dtype=torch.float32)
    if cfg.mode != "all":
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
    target_odometers = t * odometer

    # interpolate extrinsics
    target_local_ts = []
    target_indices = []
    target_extrinsics = []
    for batch_index, landmark_odometer in enumerate(landmark_odometers):
        target_index = torch.where((target_odometers >= landmark_odometer[0]) & (target_odometers <= landmark_odometer[-1]))[0]
        target_local_t = (target_odometers[target_index] - landmark_odometer[0]) / (landmark_odometer[1] - landmark_odometer[0])
        target_indices.append(target_index)
        target_local_ts.append(target_local_t)
        if target_local_t.shape[0] > 0:
            target_extrinsics.append(
                interpolate_extrinsics(
                    context_extrinsics[batch_index, 0],
                    context_extrinsics[batch_index, 1],
                    target_local_t,
                )
            )
        else:
            target_extrinsics.append(torch.empty(0, 4, 4))

    # identify frames to output for each batch
    output_indices = []
    for batch_index in range(len(target_indices) - 1):
        output_indices.append(
            target_indices[batch_index][
                target_indices[batch_index] < target_indices[batch_index + 1][0]
            ]
            if target_indices[batch_index + 1].shape[0] > 0
            else target_indices[batch_index]
        )
    output_indices.append(target_indices[-1])

    # render frames
    blend_ts_cache = defaultdict(list)
    blend_frame_cache = defaultdict(list)
    nearest_frame_cache = defaultdict(list)
    nearest_dis_cache = defaultdict(list)
    dataset.load_images = True
    output_dir = output_dir / f"{scene_id}-{args['start_frame']}-{end_frame}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir/"output"
    out = cv2.VideoWriter(
        output_file.with_suffix(".avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        dataset.cfg.image_shape[::-1],
    )
    nearst_file = output_dir/"nearest"
    nearest = cv2.VideoWriter(
        nearst_file.with_suffix(".avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        dataset.cfg.image_shape[::-1],
    )
    for target_index, target_extrinsic, target_local_t, output_index, batch in tqdm(
        zip(target_indices, target_extrinsics, target_local_ts, output_indices, dataset),
        desc="Rendering segmentation",
        total=num_batchs,
        leave=False,
    ):
        if target_extrinsic.shape[0] == 0:
            continue

        # inference and render frames
        batch = default_collate([batch])
        batch["context"] = apply_to_collection(batch["context"], Tensor, lambda x: x.to(device))
        gaussians_prob = model_wrapper.encoder(batch["context"], inference=True)["gaussians"]["gaussians"]

        target_extrinsic = target_extrinsic.to(device)
        target_extrinsic = target_extrinsic.unsqueeze(0)

        num_frames = target_extrinsic.shape[1]
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        context_extrinsics = batch["context"]["extrinsics"]
        context_images = batch["context"]["image"].cpu()
        del batch
        output_prob = model_wrapper.decoder.render_pano(
            gaussians_prob, target_extrinsic, near, far, context_extrinsics=context_extrinsics, cpu=True
        )
        batch_frame = output_prob['color'][0].cpu()
        for index, frame, local_t in zip(target_index, batch_frame, target_local_t):
            blend_frame_cache[index.item()].append(frame)
            blend_ts_cache[index.item()].append(local_t)
            context_idx = int(local_t > 1 - local_t)
            context_dis = local_t if context_idx == 0 else 1 - local_t
            nearest_frame_cache[index.item()].append(context_images[0, context_idx])
            nearest_dis_cache[index.item()].append(context_dis)

        del output_prob, gaussians_prob, target_extrinsic, near, far, context_extrinsics
        torch.cuda.empty_cache()

        # blend and output frames
        for index in output_index:
            blend_frames = torch.stack(blend_frame_cache.pop(index.item()))
            blend_ts = torch.tensor(blend_ts_cache.pop(index.item()))
            blend_distances = 0.5 - (blend_ts - 0.5).abs()
            total_weight = blend_distances.sum()
            blend_weights = blend_distances / total_weight if total_weight > 0 else torch.ones_like(blend_distances)
            blend_frames = blend_frames * blend_weights[:, None, None, None]
            blend_frame = blend_frames.sum(0)
            blend_frame = (blend_frame.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
            blend_frame = rearrange(blend_frame, "c h w -> h w c")
            blend_frame = cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)
            out.write(blend_frame)

            context_frames = torch.stack(nearest_frame_cache.pop(index.item()))
            context_dis = torch.tensor(nearest_dis_cache.pop(index.item()))
            context_frame = context_frames[context_dis.argmin()]
            context_frame = (context_frame.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
            context_frame = rearrange(context_frame, "c h w -> h w c")
            context_frame = cv2.cvtColor(context_frame, cv2.COLOR_RGB2BGR)
            nearest.write(context_frame)

    out.release()
    nearest.release()
    cmd = f"ffmpeg -i {output_file.with_suffix('.avi')} -vcodec libx264 {output_file.with_suffix('.mp4')}"
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    cmd = f"ffmpeg -i {nearst_file.with_suffix('.avi')} -vcodec libx264 {nearst_file.with_suffix('.mp4')}"
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]

    with torch.inference_mode():
        demo()

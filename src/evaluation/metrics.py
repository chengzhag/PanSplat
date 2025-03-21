from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
import numpy as np
from einops import rearrange


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    values = []
    for gt, pred in zip(ground_truth, predicted):
        value = get_lpips(predicted.device).forward(gt, pred, normalize=True)
        values.append(value)
    value = torch.cat(values)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().float().cpu().numpy(),
            hat.detach().float().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


class WSPSNR:
    """Weighted to spherical PSNR"""

    def __init__(self):
        self.weight_cache = {}

    def get_weights(self, height=1080, width=1920):
        """Gets cached weights.

        Args:
            height: Height.
            width: Width.

        Returns:
        Weights as H, W tensor.

        """
        key = str(height) + ";" + str(width)
        if key not in self.weight_cache:
            v = (np.arange(0, height) + 0.5) * (np.pi / height)
            v = np.sin(v).reshape(height, 1)
            v = np.broadcast_to(v, (height, width))
            self.weight_cache[key] = v.copy()
        return self.weight_cache[key]

    def calculate_wsmse(self, reconstructed, reference):
        """Calculates weighted mse for a single channel.

        Args:
            reconstructed: Image as B, H, W, C tensor.
            reference: Image as B, H, W, C tensor.

        Returns:
            wsmse
        """
        batch_size, height, width, channels = reconstructed.shape
        weights = torch.tensor(
            self.get_weights(height, width),
            device=reconstructed.device,
            dtype=reconstructed.dtype
        )
        weights = weights.view(1, height, width, 1).expand(
            batch_size, -1, -1, channels)
        squared_error = torch.pow((reconstructed - reference), 2.0)
        wmse = torch.sum(weights * squared_error, dim=(1, 2, 3)) / torch.sum(
            weights, dim=(1, 2, 3))
        return wmse

    def ws_psnr(self, y_pred, y_true, max_val=1.0):
        """Weighted to spherical PSNR.

        Args:
        y_pred: First image as B, H, W, C tensor.
        y_true: Second image.
        max: Maximum value.

        Returns:
        Tensor.

        """
        wmse = self.calculate_wsmse(y_pred, y_true)
        ws_psnr = 10 * torch.log10(max_val * max_val / wmse)
        return ws_psnr


class ImageMetrics:
    def __init__(self):
        self.wspsnr_calculator = WSPSNR()

    def __call__(self, rgb_gt, rgb_softmax, average=True):
        image_metrics = {}
        image_metrics['psnr'] = compute_psnr(rgb_gt, rgb_softmax).cpu().numpy()
        image_metrics['lpips'] = compute_lpips(rgb_gt, rgb_softmax).cpu().numpy()
        image_metrics['ssim'] = compute_ssim(rgb_gt, rgb_softmax).cpu().numpy()
        image_metrics['ws_psnr'] = self.wspsnr_calculator.ws_psnr(
            rearrange(rgb_softmax, "b c h w -> b h w c"),
            rearrange(rgb_gt, "b c h w -> b h w c"),
            max_val=1.0
        ).cpu().numpy()
        if average:
            image_metrics = {k: v.mean() for k, v in image_metrics.items()}
        return image_metrics


class DepthMetrics:
    def __call__(self, depth_gt, depth_est, mask):
        depth_metrics = {}
        depth_gt = depth_gt[mask]
        depth_est = depth_est[mask]
        depth_metrics['abs_rel'] = ((depth_est - depth_gt).abs() / depth_gt).mean().item()
        depth_metrics['sq_rel'] = (((depth_est - depth_gt) ** 2 / depth_gt).mean()).item()
        depth_metrics['rmse'] = torch.sqrt(((depth_est - depth_gt) ** 2).mean()).item()
        return depth_metrics

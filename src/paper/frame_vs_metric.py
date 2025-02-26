import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


output_dir = Path("outputs/frame_vs_metric")
experiments = [
    {
        "name": "PanSplat + Deferred BL",  # ls933m5x pansplat-360loc-sep_render2
        "id": "o3ejxfbz",
    },
    {
        "name": "PanSplat",  # l8l2j6pb pansplat-360loc-dis0
        "id": "irf1uj2r",
    },
    {
        "name": "MVSplat",  # 3q5jp96j mvsplat-360loc-b05
        "id": "xznsua5q",
    },
]
metric_keys = {
    "ws_psnr": "WS-PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
}


def main():
    log_root = Path("logs")
    for experiment in experiments:
        results_dir = log_root / experiment["id"] / "test" / "results.json"
        with open(results_dir, "r") as f:
            results = json.load(f)
        context_index = np.array(results["inputs"]["context_index"])[:, 0]
        target_index = np.array(results["inputs"]["target_index"])[:, 0]
        frame_dis = np.abs(context_index[:, :, None] - target_index[:, None])
        frame_dis = np.min(frame_dis, axis=1)
        metrics = {k: np.array(results["metrics"][k]).flatten() for k in metric_keys.keys()}
        experiment["frame_dis"] = frame_dis.flatten()
        experiment["metrics"] = metrics

    plot_frame_metrics(experiments, save_path=output_dir)


def plot_frame_metrics(experiments, save_path=None):
    # Plot frame distance vs metrics
    # with each metric as a separate subplot
    # and each experiment as a separate line
    # with frame distance on the x-axis
    # and the mean metric on the y-axis

    # Create subplots for each metric
    plt.rcParams.update({'font.size': 14})
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'x']
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(15, 5))

    for experiment, marker in zip(experiments, markers):
        frame_dis = experiment["frame_dis"]
        metrics = experiment["metrics"]

        for i, (metric_key, metric_name) in enumerate(metric_keys.items()):
            # Get unique frame distances and their indices
            unique_frame_dis, inverse_indices = np.unique(frame_dis, return_inverse=True)

            # Initialize array to store mean metrics for each unique frame distance
            mean_metrics = np.zeros_like(unique_frame_dis, dtype=np.float64)

            # For each unique frame distance, calculate the mean of the corresponding metric values
            for j, unique_dis in enumerate(unique_frame_dis):
                mean_metrics[j] = metrics[metric_key][inverse_indices == j].mean()

            # Plot the unique frame distances and their corresponding mean metric values
            axes[i].plot(unique_frame_dis, mean_metrics, label=experiment["name"], marker=marker, linestyle='-', alpha=0.7)
            axes[i].set_xlabel("Frame Distance")
            axes[i].set_ylabel(metric_name)
            # axes[i].set_title(f"Frame Distance vs {metric_name}")
            axes[i].legend()

    # Display the chart
    plt.tight_layout()
    if save_path:
        print(f"Saving to {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "frame_vs_metric.png")
        plt.savefig(save_path / "frame_vs_metric.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot a histogram to show the sample count per frame distance for each experiment
    fig, ax = plt.subplots(figsize=(8, 6))

    for experiment, marker in zip(experiments, markers):
        frame_dis = experiment["frame_dis"]
        # Get unique frame distances and their sample counts
        unique_frame_dis, counts = np.unique(frame_dis, return_counts=True)

        # Plot the histogram for this experiment
        ax.plot(unique_frame_dis, counts, label=experiment["name"], marker=marker, linestyle='-', alpha=0.7)

    ax.set_xlabel("Frame Distance")
    ax.set_ylabel("Sample Count")
    ax.set_title("Sample Count per Frame Distance")
    ax.legend()

    # Save the histogram plot
    if save_path:
        plt.savefig(save_path / "histogram.png")
        plt.savefig(save_path / "histogram.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

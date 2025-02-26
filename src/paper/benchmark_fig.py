import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def main():
    with open("src/paper/benchmark.yaml", "r") as f:
        config = yaml.safe_load(f)
    path = Path(config["output_dir"])
    print(f"Drawing figures from {str(path)}")
    experiments = config["experiments"]
    experiments["panogrf"] = {"name": ["PanoGRF"]}

    for key, experiment in experiments.items():
        # scan different resolutions
        output_dir = path / key
        resolution_dirs = [x for x in output_dir.glob("*/") if 'x' in x.name]
        results = []
        for resolution_dir in resolution_dirs:
            with open(resolution_dir / "info.json", "r") as f:
                result = json.load(f)
            results.append(result)
        results = sorted(results, key=lambda x: x["percentage"])

        # load results
        for result in results:
            resolution_dir = output_dir / result["key"]
            for name in ["train", "inference"]:
                file = resolution_dir / f"{name}.json"
                if not file.exists():
                    continue
                with open(file, "r") as f:
                    result[name] = json.load(f)

        experiment["results"] = results

    # analyze runtime
    experimtents_runtime = {
        key: experiments[key].copy() for key in ("panogrf", "pansplat-wo_defbp")
    }
    print("Runtime analysis at 512x1024")
    for e in experimtents_runtime.values():
        result = [r for r in e["results"] if r["height"] == 512][0]
        encoder_time = np.median(result["inference"]["execution_times"].get("encoder", [0]))
        decoder_time = np.median(result["inference"]["execution_times"]["decoder"])
        print(f"{e['name'][0]}: encoder {encoder_time:.4f}s, decoder {decoder_time:.4f}s")

    # plot gpu memory
    for figure in ["benchmark", "benchmark_ab"]:
        experiments_to_plot = {
            key: experiment.copy()
            for key, experiment in experiments.items()
            if "figure" in experiment and figure in experiment["figure"]
        }
        for e in experiments_to_plot.values():
            idx_fig = e["figure"].index(figure)
            e["name"] = e["name"][idx_fig]
        plot_memory_runtime(experiments_to_plot, path, figure)


def plot_memory_runtime(experiments, save_path, figure):
    markersize = 8
    a100_memory = 81920 / 1024  # Convert to GB
    rtx3090_memory = 24576 / 1024  # Convert to GB
    oom_offset_scale = 1

    # Predefine different colors
    colors = plt.cm.get_cmap('Set1')

    # Set font size
    plt.rcParams.update({'font.size': 12})

    # Set up the GridSpec
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    # Train and Inference memory subplots
    ax1 = fig.add_subplot(gs[0])  # Train memory vs resolution
    ax2 = fig.add_subplot(gs[1])  # Inference memory vs resolution

    for exp in experiments.values():
        mem_inference = []
        mem_train = []
        resolution_texts = []

        for result in exp["results"]:
            resolution_text = f'{result["height"]} x {result["width"]}'
            resolution_texts.append(resolution_text)
            if "inference" in result and result["inference"]["memory_stats"]:
                mem_inference.append(
                    np.median(result["inference"]["memory_stats"]["max_allocated_bytes"])
                    / (1024 ** 3)
                )  # Convert to GB
            if "train" in result and result["train"]["memory_stats"]:
                mem_train.append(
                    np.median(result["train"]["memory_stats"]["max_allocated_bytes"])
                    / (1024 ** 3)
                )

        # Train memory vs resolution
        ax1.plot(
            resolution_texts[:len(mem_train)],
            mem_train,
            label=exp["name"],
            marker=exp["marker"],
            markersize=markersize,
            color=colors(exp["color"]),
        )

        oom_count = len(resolution_texts) - len(mem_train)
        oom_offset = exp.get("oom_offset", 0) * oom_offset_scale
        if oom_count:
            ax1.plot(
                resolution_texts[len(mem_train):],
                a100_memory * np.ones(oom_count) + oom_offset,
                marker='x',
                markersize=markersize,
                color=colors(exp["color"]),
                linestyle='None',
            )

        # Inference memory vs resolution
        ax2.plot(
            resolution_texts[:len(mem_inference)],
            mem_inference,
            label=exp["name"],
            marker=exp["marker"],
            markersize=markersize,
            color=colors(exp["color"]),
        )

        oom_count = len(resolution_texts) - len(mem_inference)
        if oom_count:
            ax2.plot(
                resolution_texts[len(mem_inference):],
                a100_memory * np.ones(oom_count) + oom_offset,
                marker='x',
                markersize=markersize,
                color=colors(exp["color"]),
                linestyle='None',
            )

    # Add dashed lines and annotate RTX 3090 in the memory charts
    ax1.axhline(y=a100_memory, color=colors(6), linestyle='--')
    ax2.axhline(y=a100_memory, color=colors(6), linestyle='--')
    # ax1.axhline(y=rtx3090_memory, color=colors(8), linestyle='--')
    ax2.axhline(y=rtx3090_memory, color=colors(8), linestyle='--')
    ax1.text(
        0,
        a100_memory + 0.1,
        'A100 (80GB)',
        color=colors(6),
        fontsize=12,
        ha='left',
        va='bottom',
    )
    ax2.text(
        0,
        a100_memory + 0.1,
        'A100 (80GB)',
        color=colors(6),
        fontsize=12,
        ha='left',
        va='bottom',
    )
    ax2.text(
        0,
        rtx3090_memory + 0.1,
        'RTX 3090 (24GB)',
        color=colors(8),
        fontsize=12,
        ha='left',
        va='bottom',
    )

    # Memory vs resolution chart configuration
    ax1.set_ylabel('Training GPU Memory (GB)')
    ax2.set_ylabel('Inference GPU Memory (GB)')
    ax2.set_xlabel('Resolution')

    # Hide x-axis labels for the first subplot
    ax1.set_xticklabels([])

    # Rotate x-axis labels to avoid overlap
    for label in ax2.get_xticklabels():
        label.set_rotation(15)
        label.set_ha('right')

    # Add legend to the charts
    if figure == "benchmark":
        ax1.legend(bbox_to_anchor=(0.95, 0.9))
    ax2.legend(bbox_to_anchor=(0.45, 0.45))
    # ax2.legend(bbox_to_anchor=(0.45, 0.62))

    # Display the chart
    if save_path:
        plt.savefig(save_path / f"{figure}.png", bbox_inches='tight')
        plt.savefig(save_path / f"{figure}.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

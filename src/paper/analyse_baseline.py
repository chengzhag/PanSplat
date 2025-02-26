import json
from pathlib import Path
import numpy as np


experiments = [
    {
        "name": "Insta360",
        "id": "o3ejxfbz",
    },
    {
        "name": "360Loc",
        "id": "2v4nfmc6",
    },
]


def main():
    log_root = Path("logs")
    for experiment in experiments:
        results_dir = log_root / experiment["id"] / "test" / "results.json"
        with open(results_dir, "r") as f:
            results = json.load(f)
        extrinsics = np.array(results["inputs"]["context_extrinsics"])[:, 0]
        translations = extrinsics[..., :3, 3]
        norm = np.linalg.norm(translations[:, 0] - translations[:, 1], axis=-1)
        print(f"Mean baseline of {experiment['name']}: {norm.mean()}")


if __name__ == "__main__":
    main()

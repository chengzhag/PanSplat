import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    def __init__(self):
        self.clear_history()
        self.reset_memory_stats()

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            start_time = time()
            yield
        finally:
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(
                {
                    "execution_times": dict(self.execution_times),
                    "memory_stats": dict(self.memory_stats)
                },
                f,
                indent=4,
            )

    def summarize(self) -> None:
        if self.execution_times:
            for tag, times in self.execution_times.items():
                print(f"{tag}: {len(times)} calls, avg. {np.mean(times):.2f} seconds per call")
        if self.memory_stats:
            for tag, stats in self.memory_stats.items():
                print(f"{tag}: {len(stats)} calls, avg. {np.mean(stats) / 1024 ** 3:.2f} GB")

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)
        self.memory_stats = defaultdict(list)

    def reset_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def log_memory_stats(self) -> None:
        self.memory_stats["max_allocated_bytes"].append(torch.cuda.max_memory_allocated())
        self.memory_stats["max_cached_bytes"].append(torch.cuda.max_memory_reserved())

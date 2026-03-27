import json
import os
import time
from dataclasses import dataclass
from logging import Logger


@dataclass
class BenchmarkResult:
    rows_per_sec: float
    total_rows: int
    batch_per_sec: float
    total_batches: int
    elapsed_s: float


@dataclass
class Benchmark:
    system: str
    batch_size: int
    num_workers: int
    shuffle: bool
    results: BenchmarkResult

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            **self.results.__dict__,
        }


def measure_throughput(loader) -> BenchmarkResult:
    t0 = time.perf_counter()
    total_rows = 0
    total_batches = 0
    for batch in loader:
        total_batches += 1
        total_rows += next(iter(batch.values())).shape[0]
    elapsed = time.perf_counter() - t0

    return BenchmarkResult(
        rows_per_sec=total_rows / elapsed,
        total_rows=total_rows,
        batch_per_sec=total_batches / elapsed,
        total_batches=total_batches,
        elapsed_s=elapsed,
    )


def save_result(results: list[dict], results_dir: str, name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f)

    return path


def print_result(label: str, result: BenchmarkResult, logger: Logger) -> None:
    rows_per_sec = result.rows_per_sec
    elapsed = result.elapsed_s
    total = result.total_rows

    logger.info(
        f"  {label:<25} {rows_per_sec:>6,.0f} rows/s ({total:,} rows in {elapsed:.1f}s) {result.batch_per_sec:>6,.0f} batches/s ({result.total_batches:,} batches)"
    )

import statistics
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    rows_per_sec: float
    total_rows: int
    batch_per_sec: float
    total_batches: int
    elapsed_s: float
    rows_per_sec_std: float
    rows_per_sec_sem: float


def measure_once(loader) -> tuple[int, int, float]:
    """Iterate `loader` once, returning (total_rows, total_batches, elapsed_seconds)."""
    t0 = time.perf_counter()
    total_rows = 0
    total_batches = 0
    for batch in loader:
        total_batches += 1
        total_rows += next(iter(batch.values())).shape[0]
    elapsed = time.perf_counter() - t0
    return total_rows, total_batches, elapsed


def measure_throughput(loader, repeats: int = 5) -> BenchmarkResult:
    """Iterate `loader` `repeats` times and report mean throughput with its standard error.

    A fresh iterator is taken each repeat. Row/batch counts are deterministic across
    repeats, so only the per-run rates are aggregated. `rows_per_sec_sem` is the standard
    error of the mean (std / sqrt(n)) and is the error bar to publish; it is 0 when
    `repeats == 1`.
    """
    rows_per_sec_runs = []
    batch_per_sec_runs = []
    elapsed_runs = []
    total_rows = total_batches = 0

    for _ in range(repeats):
        total_rows, total_batches, elapsed = measure_once(loader)
        rows_per_sec_runs.append(total_rows / elapsed)
        batch_per_sec_runs.append(total_batches / elapsed)
        elapsed_runs.append(elapsed)

    std = statistics.stdev(rows_per_sec_runs) if repeats > 1 else 0.0
    return BenchmarkResult(
        rows_per_sec=statistics.mean(rows_per_sec_runs),
        total_rows=total_rows,
        batch_per_sec=statistics.mean(batch_per_sec_runs),
        total_batches=total_batches,
        elapsed_s=statistics.mean(elapsed_runs),
        rows_per_sec_std=std,
        rows_per_sec_sem=std / (repeats**0.5),
    )

import time
from dataclasses import dataclass


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

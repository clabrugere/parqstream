import argparse
import glob
import logging
import os
import sys
from datetime import datetime

from parqstream import DataLoader, Dataset
from utils import Benchmark, measure_throughput, print_result, save_result

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def configs():
    for shuffle in [False, True]:
        for batch_size in [1024, 2048, 4096]:
            for num_workers in [1, 2, 4, 6]:
                yield batch_size, num_workers, shuffle


def run_sweep(paths: list[str], results_dir: str, prefetch_factor: int, buffer_size: int) -> list[dict]:
    dataset = Dataset(paths)
    total_rows = len(dataset)
    results = []

    logger.info(f"parqstream -  ({total_rows:,} rows, {len(paths)} shards)")

    for batch_size, num_workers, shuffle in configs():
        label = f"{'shuffled' if shuffle else 'sequential'} bs={batch_size:} w={num_workers}"
        num_steps = max(total_rows // batch_size, 1)  # full epoch
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_steps=num_steps,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            buffer_size=buffer_size,
        )

        # do a full pass over the data
        warm = DataLoader(
            dataset,
            batch_size=batch_size,
            num_steps=num_steps,
            shuffle=shuffle,
            num_workers=num_workers,
            buffer_size=buffer_size,
        )
        for _ in warm:
            pass

        benchmark = Benchmark(
            system="parqstream",
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            results=measure_throughput(loader),
        )
        print_result(label, benchmark.results, logger)
        results.append(benchmark)

    save_result(
        {"bench": "throughput", "results": [r.to_dict() for r in results]},
        results_dir,
        f"bench_throughput_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/", help="Directory with .parquet shards")
    parser.add_argument("--results", default="results/", help="Output directory for JSON")
    parser.add_argument("--prefetch-factor", type=int, default=1, help="Number of buffers to prefetch")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Size of the buffer")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.parquet")))
    if not paths:
        sys.exit(f"No .parquet files found in {args.data}. Run generate_data.py first.")

    run_sweep(paths, args.results, args.prefetch_factor, args.buffer_size)

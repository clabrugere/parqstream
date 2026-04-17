import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime

from parqstream import DataLoader, Dataset
from utils import measure_throughput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BATCH_SIZES = [1024, 2048, 4096]
NUM_WORKERS = [1, 2, 4, 8]


def run(dataset: Dataset, prefetch_factor: int, buffer_size: int) -> dict:
    total_rows = len(dataset)
    results = {}

    for shuffle in [False, True]:
        mode = "shuffled" if shuffle else "sequential"
        results[mode] = {}
        for batch_size in BATCH_SIZES:
            num_steps = max(total_rows // batch_size, 1)
            base_args = {
                "batch_size": batch_size,
                "num_steps": num_steps,
                "shuffle": shuffle,
                "buffer_size": buffer_size,
            }
            row = {}
            for num_workers in NUM_WORKERS:
                for _ in DataLoader(dataset, num_workers=num_workers, **base_args):
                    pass
                comb_results = measure_throughput(
                    DataLoader(
                        dataset,
                        prefetch_factor=prefetch_factor,
                        num_workers=num_workers,
                        **base_args,
                    )
                )
                rps = round(comb_results.rows_per_sec / 1e6, 1)
                row[f"w={num_workers}"] = {**comb_results.__dict__}
                logger.info(f"- {mode} bs={batch_size} w={num_workers} -> {rps}M rows/s")

            results[mode][str(batch_size)] = row

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

    dataset = Dataset(paths)
    logger.info(f"{len(dataset):,} rows, {len(paths)} shards")
    results = run(dataset, args.prefetch_factor, args.buffer_size)

    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"bench_throughput_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"saved to {out}")

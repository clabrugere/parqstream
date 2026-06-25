import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime

from parqstream import DataLoader, Dataset
from utils import measure_once, measure_throughput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BATCH_SIZES = [1024, 2048, 4096, 8192, 16384]
NUM_WORKERS = [w for w in [1, 2, 4, 8] if w <= (os.cpu_count() or 1)]
PREFETCH_FACTORS = [1, 2, 4]


def run(dataset: Dataset, buffer_size: int, repeats: int) -> dict:
    total_rows = len(dataset)
    results = {}

    # Warmup pass
    logger.info("warmup...")
    measure_once(DataLoader(dataset, batch_size=BATCH_SIZES[-1], num_steps=max(total_rows // BATCH_SIZES[-1], 1)))

    logger.info(f"measuring throughput for {total_rows:,} rows, buffer_size={buffer_size:,}, repeats={repeats}")
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
                cell = {}
                for prefetch_factor in PREFETCH_FACTORS:
                    comb_results = measure_throughput(
                        DataLoader(
                            dataset,
                            prefetch_factor=prefetch_factor,
                            num_workers=num_workers,
                            **base_args,
                        ),
                        repeats=repeats,
                    )
                    cell[f"prefetch_factor={prefetch_factor}"] = {**comb_results.__dict__}
                    rps = round(comb_results.rows_per_sec / 1e6, 1)
                    sem = round(comb_results.rows_per_sec_sem / 1e6, 1)
                    logger.info(
                        f"- {mode} bs={batch_size} w={num_workers} pf={prefetch_factor} -> {rps}M ± {sem}M rows/s"
                    )
                row[f"w={num_workers}"] = cell

            results[mode][str(batch_size)] = row

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/", help="Directory with .parquet shards")
    parser.add_argument("--results", default="results/", help="Output directory for JSON")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Size of the buffer")
    parser.add_argument("--repeats", type=int, default=5, help="Measurement runs per cell (averaged)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.parquet")))
    if not paths:
        sys.exit(f"No .parquet files found in {args.data}. Run generate_data.py first.")

    dataset = Dataset(paths)
    logger.info(f"{len(dataset):,} rows, {len(paths)} shards")
    results = run(dataset, args.buffer_size, args.repeats)

    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"bench_throughput_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"saved to {out}")

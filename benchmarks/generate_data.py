import argparse
import logging
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def generate(num_rows: int, num_shards: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rows_per_shard = num_rows // num_shards

    for shard in range(num_shards):
        table = pa.table(
            {
                "label": pa.array(np.random.randint(0, 2, rows_per_shard, dtype=np.int32)),
                **{f"f{i}": pa.array(np.random.randn(rows_per_shard).astype(np.float32)) for i in range(10)},
            }
        )
        path = os.path.join(output_dir, f"shard_{shard:04d}.parquet")
        pq.write_table(table, path)
        logger.info(f"  wrote {path}  ({rows_per_shard:,} rows)")

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith(".parquet")
    )
    logger.info(f"\n{num_shards} shards, {num_rows:,} rows total, {total_bytes / 1e6:.1f} MB on disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark data")
    parser.add_argument("--rows", type=int, default=50_000_000, help="Total rows to generate")
    parser.add_argument("--shards", type=int, default=16, help="Number of Parquet shards")
    parser.add_argument("--output", default="data/", help="Output directory")
    args = parser.parse_args()

    logger.info(f"Generating {args.rows:,} rows across {args.shards} shards → {args.output}")
    generate(args.rows, args.shards, args.output)

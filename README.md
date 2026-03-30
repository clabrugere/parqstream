# parqstream

High-performance tabular dataloader for deep learning, backed by a Rust core.

Streams batches from one or more Parquet files (sharing the same schema) without loading everything into memory. Supports sequential reads and approximate uniform random sampling, with multi-threaded prefetching and zero-copy column transfer via the Arrow PyCapsule Interface.

## API

**`Dataset(paths, columns=None)`** — validates schemas across files and builds a global row-group index from Parquet metadata. No data is loaded at this stage. Optionally restricts to a subset of columns.

**`DataLoader(dataset, batch_size, num_steps=None, ...)`** — iterator that yields `dict[str, np.ndarray]` batches. When `num_steps` is `None` the loader cycles over the dataset indefinitely. Internally spawns:
1. A feeder thread that emits row-group chunks (optionally shuffled)
2. `num_workers` worker threads that read chunks from disk off the GIL
3. A collector thread that assembles chunks into batches of `buffer_size` rows, optionally shuffled before yielding

The batch channel has capacity `prefetch_factor`, so the next batch can be prepared while the current one is consumed. Combined row-group and buffer shuffling gives approximate uniform random sampling.

Columns are transferred to Python via Arrow PyCapsule — zero-copy for dense numeric columns, one copy for nullable or string columns.

## Usage

```python
from parqstream import Dataset, DataLoader

ds = Dataset(["part1.parquet", "part2.parquet"], columns=["a", "b"])

loader = DataLoader(
    ds,
    batch_size=256,
    num_steps=4, # will generate 4 batches
    shuffle=True, # uniform random sampling with replacement
    num_workers=4,
    prefetch_factor=2,
    buffer_size=100_000,  # rows held in memory at once
)

for batch in loader:
    a = batch["a"]  # np.ndarray, zero-copy for dense numeric columns
    b = batch["b"]
```

## Local development

Requirements: [uv](https://github.com/astral-sh/uv), Rust toolchain

```bash
git clone git@github.com:clabrugere/parqstream.git
cd parqstream
uv sync --extra dev  # install dev dependencies
maturin develop  # build Rust core and install into the venv
uv run pytest python/tests --verbose
```

## Benchmarks

Measured on a MacBook Pro M3, 50M rows across 16 Parquet shards (1 int32 + 10 float32 columns, ~2GB on disk).

**Sequential**

| batch_size | w=1 | w=2 | w=4 | w=6 |
|---|---|---|---|---|
| 1024 | 29.5M rows/s | 43.8M rows/s | 43.1M rows/s | 40.0M rows/s |
| 2048 | 30.6M rows/s | 47.1M rows/s | 59.5M rows/s | 51.3M rows/s |
| 4096 | 31.4M rows/s | 47.9M rows/s | **61.1M rows/s** | 54.3M rows/s |

**Shuffled** (rows groups and buffer shuffle)

| batch_size | w=1 | w=2 | w=4 | w=6 |
|---|---|---|---|---|
| 1024 | 28.3M rows/s | 29.3M rows/s | 27.9M rows/s | 25.5M rows/s |
| 2048 | 28.8M rows/s | 39.2M rows/s | 36.9M rows/s | 34.1M rows/s |
| 4096 | 28.6M rows/s | 43.0M rows/s | **44.1M rows/s** | 41.2M rows/s |

A first warm-up run is done over the whole data, so results are probably slightly optimistic because some low level 
caching might be happening.

To reproduce:
```bash
uv sync --extra bench
uv run benchmarks/generate_data.py --rows 50_000_000 --shards 16 --output benchmarks/data
bash benchmarks/run.sh
```

## Potential improvements
 
* Parallel file validation and global index creation
* Allow for infinite iteration by making `num_steps` an option
* Allow for epoch style iterator
* Seedable RNG

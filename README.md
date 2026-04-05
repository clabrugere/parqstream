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
    num_steps=None, # will cycle through the dataset
    shuffle=True, # uniform random sampling
    num_workers=4,
    prefetch_factor=2,
    buffer_size=100_000, # rows held in memory at once,
    seed=42 # yield rows in the same order between two runs when shuffling is active
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

| batch_size | w=1 | w=2 | w=4 | w=8 |
|---|---|---|---|---|
| 1024 | 37.7M rows/s | 48.6M rows/s | 47.5M rows/s | 39.7M rows/s |
| 2048 | 38.3M rows/s | 56.7M rows/s | 69.7M rows/s | 55.9M rows/s |
| 4096 | 38.3M rows/s | 57.3M rows/s | **70.1M rows/s** | 56.7M rows/s |

**Shuffled** (rows groups and buffer shuffle)

| batch_size | w=1 | w=2 | w=4 | w=8 |
|---|---|---|---|---|
| 1024 | 29.8M rows/s | 29.4M rows/s | 28.8M rows/s | 25.2M rows/s |
| 2048 | 36.1M rows/s | 39.0M rows/s | 38.1M rows/s | 32.0M rows/s |
| 4096 | 36.3M rows/s | 46.8M rows/s | **45.2M rows/s** | 36.3M rows/s |

A first warm-up run is done over the whole data, so results are probably slightly optimistic because some low level 
caching might be happening.

**GPU training** (NVIDIA A10G, 10M rows across 16 shards, 100 float32 features, 10 classes)

135M-parameter MLP (100→8192→8192→8192→10), batch size 65,536, 8 workers, shuffled.

| metric | value |
|---|---|
| rows/s | 26,800 |
| GPU utilization | 99.98% |
| VRAM used | 12.8 / 24 GB |
| data pipeline overhead | 0.9% |
| compute time / step | 2.42s |

The data pipeline accounts for less than 1% of step time showing that the loader keeps the GPU fully saturated.

To reproduce:
```bash
uv sync --extra bench
uv run benchmarks/generate_data.py --rows 50_000_000 --shards 16 --output benchmarks/data
bash benchmarks/run.sh
```

## Build
 
If you want to compile targeting an architecture (for example x86_64 linux):

```bash
uv sync --extra build_wheel
rustup target add x86_64-unknown-linux-gnu
maturin build --release --target x86_64-unknown-linux-gnu --zig
```

## Potential improvements

* Support reading from remote storages
* Support distributed training
* Parallel file validation and global index creation
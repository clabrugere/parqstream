<br />
<p align="center">
    <a href="https://github.com/clabrugere/parqstream#gh-light-mode-only" class="only-light">
      <img src="./assets/parqstream-light.png" width="50%"/>
    </a>
    <a href="https://github.com/clabrugere/parqstream#gh-dark-mode-only" class="only-dark">
      <img src="./assets/parqstream-dark.png" width="50%"/>
    </a>
</p>

<h3><p align="center">High-performance tabular dataloader for deep learning</p></h3>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20–%203.13-blue" />
  <img alt="License" src="https://img.shields.io/github/license/clabrugere/parqstream" />
</p>

---

parqstream streams batches directly from Parquet files without loading the full dataset into memory. It is built in Rust with Python bindings, and designed to keep your GPU saturated while consuming constant memory regardless of dataset size.

- **Zero-copy transfer**: dense numeric columns are transferred to Python via the Arrow PyCapsule Interface with no data copy
- **Multi-threaded prefetching**: worker threads read and decode chunks off the GIL while the main thread consumes the current buffer
- **Approximate uniform shuffling**: combined row-group ordering and buffer shuffling without loading the full dataset
- **Exact checkpointing**: capture and restore the iteration position to the row, including across restarts
- **Distributed training**: partition the dataset across processes with `rank` / `world_size`; each rank gets a disjoint subset per epoch, with shuffle and checkpointing fully supported
- **Framework-agnostic**: yields `dict[str, np.ndarray]` by default; pass a `collate_fn` for PyTorch tensors or any other format

---

## Installation

Requires the Rust toolchain and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/clabrugere/parqstream.git
cd parqstream
uv sync
maturin develop --release
```

---

## Quickstart

```python
from parqstream import Dataset, DataLoader

ds = Dataset(["part1.parquet", "part2.parquet"], columns=["feature", "label"])

loader = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=4, seed=42)

for batch in loader:
    features = batch["feature"]  # np.ndarray, zero-copy for dense numeric
    labels   = batch["label"]
    train(features, labels)
```

---

## Usage

### Sequential loading

```python
from parqstream import Dataset, DataLoader

ds = Dataset(
    paths=["shard_00.parquet", "shard_01.parquet"],
    columns=["x", "y"],  # omit to load all columns
)

loader = DataLoader(
    ds,
    batch_size=2048,
    num_workers=4,
)

for batch in loader:
    x = batch["x"]  # np.ndarray
    y = batch["y"]
```

### Shuffled, indefinite cycling

Pass `num_steps=None` (the default) to cycle the dataset indefinitely. Useful for training loops that count steps rather than epochs. Set `shuffle=True` for approximate uniform random sampling: row groups are visited in a random order each epoch, and each buffer is shuffled before being sliced into batches.

```python
loader = DataLoader(
    ds,
    batch_size=4096,
    num_steps=None,       # cycle indefinitely
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,    # prepare the next buffer while consuming the current one
    buffer_size=500_000,  # rows held in memory at once
    seed=42,
)

for batch in loader:
    ...
```

### PyTorch integration

Pass a `collate_fn` to convert the raw `Batch` to framework tensors. The function receives a `Batch` object and its return value is yielded directly, bypassing the default numpy conversion.

```python
import torch
import pyarrow as pa
from parqstream import Dataset, DataLoader

def to_torch(batch):
    return {
        col.name: torch.from_numpy(pa.array(col).to_numpy(zero_copy_only=False))
        for col in batch.columns()
    }

ds = Dataset(["train.parquet"])
loader = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=4, collate_fn=to_torch)

for batch in loader:
    logits = model(batch["x"])  # torch.Tensor
```

### Distributed training

Use `rank` and `world_size` to partition the dataset across processes. Each rank iterates over a disjoint subset of row groups per epoch; together all ranks cover the full dataset exactly once. Shuffle and checkpointing work identically to the single-process case.

```python
import torch.distributed as dist
from parqstream import Dataset, DataLoader

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

ds = Dataset(["train.parquet"])
loader = DataLoader(
    ds,
    batch_size=4096,
    num_steps=10_000,
    shuffle=True,
    seed=42,
    rank=rank,
    world_size=world_size,
)

for batch in loader:
    train(batch)
```

The dataset is partitioned at the row-group level using a strided assignment: rank `r` takes positions `r, r+W, r+2W, …` from the epoch's visit order. This handles uneven splits (`num_row_groups % world_size ≠ 0`) without dropping data. `world_size` must not exceed the number of row groups so that every rank receives at least one.

Checkpointing works per-rank. Save and restore each rank's loader independently, using the same `rank` and `world_size`:

```python
# save
torch.save({"model": model.state_dict(), "loader": loader.state_dict()}, f"ckpt_rank{rank}.pt")

# restore
state = torch.load(f"ckpt_rank{rank}.pt")
model.load_state_dict(state["model"])
loader.load_state_dict(state["loader"])
```

---

### Checkpointing

Call `checkpoint()` at any point during iteration to capture the exact position, then resume on a new loader with `load_checkpoint()`. The same sequence of batches is reproduced from that point forward, as long as the loader is created with the same parameters and seed.

```python
from parqstream import Dataset, DataLoader

ds = Dataset(["train.parquet"])
loader = DataLoader(ds, batch_size=4096, num_steps=10_000, shuffle=True, seed=42)

for i, batch in enumerate(loader):
    train(batch)
    if i == 4999:
        cp = loader.checkpoint()
        break

# resume from step 5000
new_loader = DataLoader(ds, batch_size=4096, num_steps=10_000, shuffle=True, seed=42)
new_loader.load_checkpoint(cp)

for batch in new_loader:
    train(batch)
```

`state_dict()` / `load_state_dict()` are convenience wrappers that serialize the checkpoint to a plain Python dict, suitable for saving alongside model weights:

```python
import torch

# save
torch.save(
    {"model": model.state_dict(), "loader": loader.state_dict()},
    "checkpoint.pt",
)

# restore
state = torch.load("checkpoint.pt")
model.load_state_dict(state["model"])
loader.load_state_dict(state["loader"])
```

---

## API reference

### `Dataset(paths, columns=None)`

Reads Parquet metadata from all files, validates that the requested columns are present and type-compatible across all files, and builds a global row-group index. No row data is loaded at this stage.

| Parameter | Type | Description |
|-----------|------|-------------|
| `paths` | `list[str]` | Paths to one or more Parquet files. All files must contain the requested columns with compatible types. |
| `columns` | `list[str] \| None` | Column names to project. `None` loads all columns. |

| Property | Description |
|----------|-------------|
| `columns` | List of column names in the dataset. |
| `num_files` | Number of source files. |
| `num_row_groups` | Total row groups across all files. |
| `__len__()` | Total number of rows across all files. |

---

### `DataLoader(dataset, batch_size, ...)`

Iterator that yields batches. Internally spawns a multi-threaded pipeline: a job dispatcher thread, `num_workers` reader threads (off the GIL), and a buffer assembler thread (concatenates chunks and shuffles). The main thread receives assembled buffers, stitches any unconsumed tail rows to the front, and slices into batches with the GIL released.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `Dataset` | / | Source dataset. |
| `batch_size` | `int` | / | Number of rows per batch. |
| `num_steps` | `int \| None` | `None` | Total batches to yield. `None` cycles indefinitely. |
| `shuffle` | `bool` | `False` | Shuffle row-group visit order and each buffer for approximate uniform random sampling. |
| `num_workers` | `int` | `4` | Number of parallel reader threads. |
| `prefetch_factor` | `int` | `1` | Number of assembled buffers queued ahead of consumption. Higher values overlap I/O and assembly with the main thread at the cost of memory. |
| `buffer_size` | `int \| None` | total rows | Rows accumulated before slicing into batches. Controls memory usage and shuffle quality. |
| `seed` | `int \| None` | `None` | RNG seed for reproducible shuffling and checkpointing. Required to resume from a checkpoint. |
| `collate_fn` | `callable \| None` | `None` | Function `(Batch) -> any`. When set, its return value is yielded instead of `dict[str, np.ndarray]`. |
| `rank` | `int` | `0` | This process's rank in a distributed setup (0-based). Together with `world_size`, selects which row groups this loader iterates over. |
| `world_size` | `int` | `1` | Total number of processes. The dataset is partitioned into `world_size` disjoint subsets via strided assignment. Must not exceed the number of row groups. |

| Method | Description |
|--------|-------------|
| `checkpoint()` | Returns a `Checkpoint` capturing the current iteration position. |
| `load_checkpoint(cp)` | Restores iteration state from a `Checkpoint`. Must be called before the first `__next__`. |
| `state_dict()` | Returns the checkpoint as a plain Python dict. |
| `load_state_dict(d)` | Restores state from a dict produced by `state_dict()`. |

---

### `Batch`

A single batch of rows, backed by an Arrow RecordBatch. Implements the Arrow PyCapsule Interface (`__arrow_c_stream__`) for framework integration.

| Method | Description |
|--------|-------------|
| `columns()` | Returns all `Column` objects in schema order. |
| `column(name)` | Returns the named `Column`. Raises `KeyError` if not found. |
| `__len__()` | Number of rows in this batch. |

---

### `Column`

A single named column within a `Batch`, backed by an Arrow array. Implements `__arrow_c_array__` for zero-copy transfer to compatible libraries.

| Property/Method | Description |
|-----------------|-------------|
| `name` | Column name. |
| `__arrow_c_array__()` | Returns `(schema_capsule, array_capsule)`. Zero-copy for dense numeric types (`float32`, `float64`, `int32`, …); one copy for nullable or string columns. |

---

### `Checkpoint`

Opaque object returned by `DataLoader.checkpoint()`. Captures training epoch, stream epoch, row-group position, buffer offset, and steps remaining.

| Method | Description |
|--------|-------------|
| `to_dict()` | Serialize to a plain Python dict. |
| `Checkpoint.from_dict(d)` | Deserialize from a dict produced by `to_dict()`. |

---

## Architecture

```
Dataset ──► DataLoader
               │
               ├── job_dispatcher    emits jobs (read task + result channel) optionally shuffled per epoch
               │
               ├── worker × N        reads and decodes chunks from disk, off the GIL
               │
               ├── buffer_builder    assembles chunks (pulled in dispatch order via per-job channels) into a buffer of buffer_size rows; shuffles it
               │                     ↕ prefetch_factor assembled buffers queued ahead
               └── Buffer            stitches unconsumed tail rows to the front, slices into batch_size batches, off the GIL
```

**Ordering guarantee.** Workers finish jobs in arbitrary order, but `buffer_builder` always consumes chunks in dispatch order. This is structurally enforced: `job_dispatcher` creates a single-use `bounded(1)` channel per job and immediately enqueues the receiver into a FIFO `pending` channel before any worker touches the job. `buffer_builder` drains `pending` in order, blocking on each job's dedicated receiver before moving to the next. A fast worker completing job N+1 before job N simply deposits its result into N+1's channel, where it waits until N has been consumed.

Columns are returned to Python via the Arrow PyCapsule Interface: zero-copy for dense numeric types, one copy otherwise.

---

## Benchmarks

Measured on a MacBook Pro M3, 50M rows across 16 Parquet shards (1 int32 + 10 float32 columns, ~2 GB on disk), `buffer_size=1_000_000`. Each cell is the `mean ± SE` over 5 runs (M rows/s) after a warmup pass.

**Sequential**

| batch_size | prefetch | 1 worker | 2 workers | 4 workers | 8 workers |
|:----------:|:--:|:--------:|:---------:|:---------:|:---------:|
| 1024  | 1 | 35.3 ± 1.4 | 49.8 ± 0.2 | 47.2 ± 0.4 | 39.6 ± 0.4 |
|       | 2 | 36.8 ± 0.2 | 49.7 ± 0.2 | 47.6 ± 0.3 | 38.4 ± 0.3 |
|       | 4 | 36.7 ± 0.2 | 49.9 ± 0.1 | 47.2 ± 0.2 | 36.5 ± 0.6 |
| 2048  | 1 | 36.1 ± 0.2 | 57.5 ± 0.6 | 74.5 ± 1.2 | 51.0 ± 2.0 |
|       | 2 | 37.3 ± 0.2 | 57.9 ± 0.1 | 73.0 ± 0.3 | 56.7 ± 0.7 |
|       | 4 | 37.4 ± 0.1 | 57.8 ± 0.3 | 61.5 ± 2.6 | 57.8 ± 1.0 |
| 4096  | 1 | 36.1 ± 0.3 | 55.9 ± 0.7 | 71.3 ± 1.2 | 59.7 ± 1.0 |
|       | 2 | 36.5 ± 0.2 | 55.4 ± 1.2 | 72.9 ± 0.6 | 59.9 ± 0.3 |
|       | 4 | 36.6 ± 0.3 | 55.6 ± 0.6 | 72.3 ± 0.5 | 60.8 ± 0.3 |
| 8192  | 1 | 36.9 ± 0.2 | 57.7 ± 0.6 | 73.5 ± 0.3 | 59.1 ± 0.4 |
|       | 2 | 37.6 ± 0.1 | 57.7 ± 0.3 | 74.8 ± 0.8 | 58.8 ± 0.8 |
|       | 4 | 37.5 ± 0.2 | 57.6 ± 0.4 | 75.2 ± 1.1 | 60.7 ± 0.4 |
| 16384 | 1 | 37.0 ± 0.2 | 58.3 ± 0.3 | 73.9 ± 1.4 | 62.2 ± 0.2 |
|       | 2 | 37.4 ± 0.1 | 58.7 ± 0.2 | **75.7 ± 0.3** | 61.0 ± 0.2 |
|       | 4 | 37.3 ± 0.1 | 57.9 ± 0.2 | 74.1 ± 0.7 | 61.4 ± 0.6 |

**Shuffled** (row-group order + buffer shuffle)

| batch_size | prefetch | 1 worker | 2 workers | 4 workers | 8 workers |
|:----------:|:--:|:--------:|:---------:|:---------:|:---------:|
| 1024  | 1 | 35.0 ± 0.2 | 48.4 ± 0.1 | 45.2 ± 0.4 | 37.6 ± 0.1 |
|       | 2 | 35.6 ± 0.2 | 48.4 ± 0.2 | 45.2 ± 0.2 | 37.6 ± 0.2 |
|       | 4 | 35.3 ± 0.2 | 48.4 ± 0.2 | 45.3 ± 0.3 | 37.3 ± 0.3 |
| 2048  | 1 | 35.3 ± 0.2 | 53.2 ± 0.1 | 55.7 ± 0.5 | 43.1 ± 0.6 |
|       | 2 | 35.2 ± 0.2 | 53.7 ± 0.5 | 55.2 ± 0.4 | 39.9 ± 1.5 |
|       | 4 | 35.6 ± 0.2 | 53.0 ± 0.4 | 53.9 ± 0.6 | 42.4 ± 1.1 |
| 4096  | 1 | 35.7 ± 0.1 | 51.8 ± 0.3 | 54.3 ± 0.7 | 39.2 ± 0.8 |
|       | 2 | 32.6 ± 1.8 | 52.5 ± 0.2 | 52.2 ± 0.7 | 38.8 ± 1.0 |
|       | 4 | 33.5 ± 0.3 | 52.8 ± 0.3 | 46.5 ± 2.4 | 36.0 ± 2.7 |
| 8192  | 1 | 34.0 ± 0.5 | 54.5 ± 0.7 | 54.5 ± 0.7 | 44.8 ± 0.7 |
|       | 2 | 36.2 ± 0.1 | 51.5 ± 0.9 | 54.3 ± 1.5 | 43.7 ± 0.2 |
|       | 4 | 36.5 ± 0.1 | 49.0 ± 1.5 | 55.1 ± 0.4 | 43.4 ± 0.5 |
| 16384 | 1 | 35.5 ± 0.2 | 52.9 ± 0.7 | 55.4 ± 0.7 | 37.6 ± 2.1 |
|       | 2 | 35.6 ± 0.2 | 54.3 ± 0.3 | **56.3 ± 0.3** | 37.0 ± 1.7 |
|       | 4 | 35.7 ± 0.3 | 53.5 ± 0.3 | 52.8 ± 2.1 | 42.7 ± 1.2 |

**GPU training** (NVIDIA A10G, 10M rows x 16 shards, 100 float32 features, 10 classes)

135M-parameter MLP (100 → 8192 → 8192 → 8192 → 10), batch size 65 536, 8 workers, shuffled.

| metric | value |
|--------|-------|
| rows/s | 26 800 |
| GPU utilization | 99.98% |
| VRAM used | 12.8 / 24 GB |
| data pipeline overhead | **0.9%** |
| compute time per step | 2.42 s |

The data pipeline accounts for less than 1% of step time, the GPU stays fully saturated.

To reproduce:

```bash
uv sync --extra bench
uv run benchmarks/generate_data.py --rows 50_000_000 --shards 16 --output benchmarks/data
bash benchmarks/run.sh
```

---

## Development

Requirements: [uv](https://github.com/astral-sh/uv), Rust toolchain.

```bash
git clone https://github.com/clabrugere/parqstream.git
cd parqstream
uv sync --extra dev        # install dev dependencies
maturin develop            # compile Rust core and install into the venv
uv run pytest python/tests --verbose
```

To build a release wheel targeting a specific architecture (e.g. x86-64 Linux):

```bash
uv sync --extra build_wheel
rustup target add x86_64-unknown-linux-gnu
maturin build --release --target x86_64-unknown-linux-gnu --zig
```

---

## Roadmap

- Remote storage (S3, GCS, HTTPS) via presigned URLs with automatic refresh
- Parallel file validation and index construction at startup

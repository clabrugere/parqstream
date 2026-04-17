<br />
<p align="center">
    <a href="https://github.com/clabrugere/parqstream#gh-light-mode-only" class="only-light">
      <img src="./assets/parqstream-light.png" width="50%"/>
    </a>
    <a href="https://github.com/clabrugere/parqstream#gh-dark-mode-only" class="only-dark">
      <img src="./assets/parqstream-dark.png" width="50%"/>
    </a>
</p>

<h2><p align="center">High-performance tabular dataloader for deep learning</p></h2>


Stream batches from Parquet files without loading everything into memory. Supports sequential reads and approximate uniform random sampling, with multi-threaded prefetching and zero-copy column transfer via the Arrow PyCapsule Interface.

## API

**`Dataset(paths, columns=None)`** — validates schemas across files and builds a global row-group index from Parquet metadata. No data is loaded at this stage. Optionally restricts to a subset of columns.

**`DataLoader(dataset, batch_size, num_steps=None, ...)`** — iterator that yields `dict[str, np.ndarray]` batches. When `num_steps` is `None` the loader cycles over the dataset indefinitely. Internally spawns:
1. A feeder thread that emits row-group chunks (optionally shuffled)
2. `num_workers` worker threads that read chunks from disk off the GIL
3. A collector thread that assembles chunks into fills of `buffer_size` rows
4. The main thread reads fills via a buffer that optionally shuffles each fill and slices it into `batch_size` batches; any unconsumed rows from the previous fill are prepended so batches never straddle a fill boundary with a gap

The fill channel has capacity `prefetch_factor`, so the next fill can be prepared while the current one is consumed. Combined row-group and buffer shuffling gives approximate uniform random sampling.

Columns are transferred to Python via Arrow PyCapsule — zero-copy for dense numeric columns, one copy for nullable or string columns.

**`Checkpoint`** — opaque object returned by `DataLoader.checkpoint()` that captures the exact iteration position (epoch, row-group offset, buffer offset, steps remaining). Supports `to_dict()` / `from_dict()` for serialization.

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

To produce framework tensors instead of numpy arrays, pass a `collate_fn`. It receives the raw `Batch` object and its return value is yielded as-is, bypassing the default numpy conversion:

```python
import torch
import pyarrow as pa
from parqstream import Dataset, DataLoader

def collate_fn(batch):
    return {col.name: torch.from_numpy(pa.array(col).to_numpy(zero_copy_only=False)) for col in batch.columns()}

ds = Dataset(["part1.parquet", "part2.parquet"], columns=["a", "b"])

loader = DataLoader(ds, batch_size=256, collate_fn=collate_fn)

for batch in loader:
    a = batch["a"]  # torch.Tensor
    b = batch["b"]
```

### Checkpointing

Call `checkpoint()` at any point during iteration to capture the exact position in the dataset, then pass it to `load_checkpoint()` on a new loader to resume:

```python
from parqstream import Dataset, DataLoader

ds = Dataset(["part.parquet"])
loader = DataLoader(ds, batch_size=256, num_steps=1000, shuffle=True, seed=42)

for i, batch in enumerate(loader):
    train(batch)
    if i == 499:
        cp = loader.checkpoint()
        break

# resume from step 500
new_loader = DataLoader(ds, batch_size=256, num_steps=1000, shuffle=True, seed=42)
new_loader.load_checkpoint(cp)
for batch in new_loader:
    train(batch)
```

`state_dict()` and `load_state_dict()` are convenience wrappers that convert the checkpoint to and from a plain Python dict, suitable for saving alongside a model checkpoint:

```python
import torch

state = {
    "model": model.state_dict(),
    "loader": loader.state_dict(),
}
torch.save(state, "checkpoint.pt")

# restore
state = torch.load("checkpoint.pt")
model.load_state_dict(state["model"])
loader.load_state_dict(state["loader"])
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

## Improvements

* Support reading from remote storages
* Support distributed training
* Parallel file validation and global index creation
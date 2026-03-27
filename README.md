# parqstream

Stream parquet rows to your machine learning models.

## Overview

This small library allows to stream batches from one or multiple parquet files (sharing the same schema) without
loading everything in memory. It supports sequential read or uniform random sampling as well as parametrizable 
multi-threaded prefetching and zero-copy when possible.

The core engine is implemented in rust and a simple API is exposed to the python consumer through two objects:
* `Dataset` storing paths of parquet files, validating schemas and building a global index using file's metadata for random access.
* `Dataloader` to iterate over (optionally randomly sampled) batches over some columns, with prefetch and multi-threaded batch 
construction without locking Python's Global Interpreter. The iteration stops when `num_steps` batches have been returned.

## How it works

### Dataset

`Dataset` first validates the files' schema before building a global index of `row_groups`. The index is an array of 
`RowGroupMeta` that stores a local row group index (within a file) as well as an offset corresponding to the 
index of the first row of a row group in the global flattened dataset. At this point no data is loaded but only the
parquet files' metadata required to build the index and selecting rows during batch creation.

It exposes a method to read a slice within a given `row_group`

### Dataloader

`Dataloader` is an iterator bounded by `num_steps` that returns a `Batch` each time. On initialization, the iterator
spawn a pool of threads:
1. a thread that emit chunks of rows groups to a chunk channel (feeder), from optionally shuffled row groups.
2. `num_workers` threads that read chunks from the chunk channel, load them from disk and send to a data channel (workers)
3. a thread that collects chunks from the data channel until it has > `buffer_size` rows, concatenates them and sends to a batch channel (collector)

The buffer is optionally shuffled such that with the row group shuffling, uniform random sampling is approximated.
The buffer channel has a capacity `prefetch_factor` such that another buffer can be prepared while the current buffer is being consumed, smoothing throughput.

### Batch

It is a light wrapper around an arrow array's `RecordBatch` containing the data. `Columns` are transmitted to the 
python interpreter using a `PyCapsule` to avoid copying the data. On the python side, this is transformed to a dict of 
numpy's ndarrays without copy when possible.

### Notes

* For epoch style training, one must set `num_steps = len(df) / batch_size`

## Usage

```python
parquet_files = [...]
ds = Dataset(parquet_files, columns=["a", "b"]) # optionally select subset of columns
loader = DataLoader(
    ds, 
    batch_size=256,
    num_steps=4, # will generate 4 batches
    shuffle=True, # uniform random sampling with replacement
    num_workers=4,
    prefetch_factor =2,
)

for batch in loader:
    a = batch["a"] # numpy array as a view over an arrow array
    b = batch["b"]

    ...
```


## Potential improvements
 
* Parallel file validation and global index creation
* Allow for infinite iteration by making `num_steps` an option
* Allow for epoch style iterator
* Seedable RNG

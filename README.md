# parqstream

Stream parquet rows to your machine learning models.

## Overview

This small library allows to stream batches from one or multiple parquet files (sharing the same schema) without
loading everything in memory. It supports sequential read or uniform random sampling (with replacement) as well 
as parametrizable multi-threaded prefetching and zero-copy when possible.

The core engine is implemented in rust and a simple API is exposed to the python consumer through two objects:
* `Dataset` storing paths of parquet files, and building a global index using file's metadata for random access
* `Dataloader` to iterate over (randomly sampled) batches over some columns, with prefetch and multi-threaded batch 
construction without locking the GIL. The iteration stops when `num_steps` batches have been returned.

## How it works

### Dataset

`Dataset` first validates the files' schema then builds a global index of `row_groups`. The index is an array of 
`RowGroupMeta` that stores a local row group index (within a file) as well as an offset corresponding to the 
index of the first row of a row group in the global flattened dataset. At this point no data is loaded, only the
parquet files' metadata required to build the index. This allows us to index over the entire dataset using the global
index of a row.

To locate a row with its global index, we perform a binary search over the `row_groups` to return the latest row group whose offset is
before the row. Once we have located the row group containing the row, it's just a matter of indexing it using 
the offset and the row's global index.

### Dataloader

`Dataloader` is an iterator bounded by `num_steps` that returns a `Batch` each time. On initialization, the iterator
spawn a pool of threads:
* one thread to generate batch indices, sent through a crossbeam's `bounded` channel for backpressure,
* `num_workers` threads doing the heavy work. They wait for indices coming from the index channel and start loading data
as soon as they receive them. All those threads send batches through a single batch channel, that is then forwarded to
the python interpreter.

### Batch

It is a light wrapper around an arrow array's `RecordBatch` containing the data. `Columns` are transmitted to the 
python interpreter using a `PyCapsule` to avoid copying the data. On the python side, this is transformed to a dict of numpy ndarrays
without copy when possible.

### Notes

* Sampling with replacement is performed, and duplicate rows in a batch are dropped. It means that batches can have a smaller size than `batch_size` when it happens.
* Row order is not preserved because they are returned in the order of the files.
* In sequential mode, steps can go back to the start of the dataset if `num_steps` > `total_rows`.
* For epoch style training, one must set `num_steps = len(df) / batch_size`

## Usage

```python
parquet_files = [...]
ds = Dataset(parquet_files)
loader = DataLoader(
    ds, 
    batch_size=256, 
    num_steps=4, 
    columns=["a", "b"]
    shuffle=True,
    num_workers=4,
    prefetch_factor =4,
)

for batch in loader:
    a = batch["a"] # numpy array as a view over an arrow array
    b = batch["b"]

    ...
```


## Potential improvements
 
* Parallel file validation
* Parallel global index creation
* Buffer for sampling without replacement
* Allow for infinite iteration by making `num_steps` an option
* Allow for epoch style training
* File metadata cache to avoid open-header read on each batch

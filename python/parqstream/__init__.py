from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pyarrow as pa

from parqstream._parqstream import Column
from parqstream._parqstream import DataLoader as _RustDataLoader
from parqstream._parqstream import Dataset as _RustDataset

__all__ = ["Dataset", "DataLoader"]


def _col_to_numpy(col: Column) -> np.ndarray:
    name = getattr(col, "name", "<unknown>")
    array = pa.array(col)
    try:
        return array.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        warnings.warn(f"Column '{name}' cannot be zero-copied; falling back to one-copy conversion", UserWarning)
        return array.to_numpy(zero_copy_only=False)


class DataLoader:
    """Prefetching dataloader that yields batches as `dict[str, np.ndarray]`.

    Backed by Rust workers that read and assemble Arrow batches off the GIL.
    Columns are transferred via the Arrow PyCapsule Interface — zero-copy for
    dense numeric columns, one copy for nullable or string columns.

    Args:
        dataset: Source dataset.
        batch_size: Number of rows per batch.
        num_steps: Total number of batches to yield. If `None` (default), the
            loader cycles over the dataset indefinitely.
        shuffle: If `True`, shuffles row-group order and buffer contents for
            approximate uniform random sampling.
        num_workers: Number of parallel reader threads.
        prefetch_factor: Capacity of the batch channel; higher values pipeline
            more batches ahead of consumption.
        buffer_size: Number of rows to accumulate before slicing into batches.
            If `None`, each row group is yielded as-is.
        seed: Random seed for reproducible shuffling. Has no effect when
            `shuffle` is `False`.
        collate_fn: Optional callable that receives the raw `Batch` object and
            returns whatever the iteration protocol should yield. When provided,
            the default numpy conversion is bypassed entirely, so the callable
            is responsible for all column extraction and type conversion.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_steps: int | None = None,
        shuffle: bool = False,
        num_workers: int = 1,
        prefetch_factor: int = 1,
        buffer_size: int | None = None,
        seed: int | None = None,
        collate_fn: Callable | None = None,
    ) -> None:
        self._dataloader = _RustDataLoader(
            dataset._dataset,
            batch_size,
            num_steps,
            shuffle,
            num_workers,
            prefetch_factor,
            buffer_size,
            seed,
        )
        self._collate_fn = collate_fn

    def __iter__(self) -> DataLoader:
        self._dataloader.__iter__()
        return self

    def __next__(self) -> dict[str, np.ndarray] | Any:
        batch = next(self._dataloader)

        if self._collate_fn is not None:
            return self._collate_fn(batch)

        return {col.name: _col_to_numpy(col) for col in batch.columns()}

    def __len__(self) -> int:
        return len(self._dataloader)

    def __repr__(self) -> str:
        return repr(self._dataloader)


class Dataset:
    """Dataset distributed over one or more Parquet files only read as needed.

    Parquet metadata (schema, row-group statistics) is read eagerly at
    construction time; column data is read lazily by the dataloader workers.

    Args:
        paths: List of Parquet file paths to include in the dataset.
        columns: Optional list of column names to project. If `None`, all
            columns are read.
    """

    def __init__(self, paths: list[str], columns: list[str] | None = None) -> None:
        self._dataset = _RustDataset(paths, columns)

    @property
    def columns(self) -> list[str]:
        return self._dataset.columns

    @property
    def num_files(self) -> int:
        return self._dataset.num_files

    @property
    def num_row_groups(self) -> int:
        return self._dataset.num_row_groups

    def __len__(self) -> int:
        return len(self._dataset)

    def __repr__(self) -> str:
        return repr(self._dataset)

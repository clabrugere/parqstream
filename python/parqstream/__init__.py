from __future__ import annotations

import numpy as np
import pyarrow as pa

from parqstream._parqstream import Column
from parqstream._parqstream import DataLoader as _RustDataLoader
from parqstream._parqstream import Dataset as _RustDataset

__all__ = ["Dataset", "DataLoader"]


def _col_to_numpy(col: Column) -> np.ndarray:
    return pa.array(col).to_numpy(zero_copy_only=False)


class DataLoader:
    """Prefetching dataloader that yields batches as `dict[str, np.ndarray]`.

    Backed by Rust workers that read and assemble Arrow batches off the GIL.
    Columns are transferred via the Arrow PyCapsule Interface — zero-copy for
    dense numeric columns, one copy for nullable or string columns.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_steps: int,
        shuffle: bool = False,
        num_workers: int = 4,
        prefetch_factor: int = 4,
    ) -> None:
        self._dataloader = _RustDataLoader(
            dataset._dataset,
            batch_size,
            num_steps,
            shuffle,
            num_workers,
            prefetch_factor,
        )

    def __iter__(self) -> DataLoader:
        iter(self._dataloader)
        return self

    def __next__(self) -> dict[str, np.ndarray]:
        batch = next(self._dataloader)
        return {name: _col_to_numpy(batch.column(name)) for name in batch.columns}

    def __repr__(self) -> str:
        return repr(self._dataloader)


class Dataset:
    """Dataset representing distributed over one or more Parquet files only read as needed."""

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

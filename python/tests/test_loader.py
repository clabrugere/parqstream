import numpy as np
import pytest
import torch
from parqstream import DataLoader, Dataset


def test_basic_iteration(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(
        ds,
        batch_size=256,
        num_steps=4,
        columns=["f1", "label"],
        num_workers=2,
        prefetch_factor=2,
    )

    assert sum(1 for _ in loader) == 4

    for batch in loader:
        assert set(batch.keys()) == {"f1", "label"}
        assert batch["f1"].dtype == np.float32
        assert batch["label"].dtype == np.int32
        assert len(batch["f1"]) > 0


def test_batch_size(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=512, num_steps=3, columns=["f1"])

    for batch in loader:
        # Each batch should have at most batch_size rows but potentially fewer it it had duplicate rows after sampling
        assert len(batch["f1"]) <= 512
        assert len(batch["f1"]) > 0


def test_all_columns_default(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=128, num_steps=4)

    for batch in loader:
        assert set(batch.keys()) == set(ds.columns)


def test_multi_file(two_parquet_paths):
    ds = Dataset(two_parquet_paths)
    loader = DataLoader(ds, batch_size=256, num_steps=4, columns=["f1", "label"])

    assert sum(1 for _ in loader) == 4


def test_multiple_iterations(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=256, num_steps=4, columns=["f1"])

    assert sum(1 for _ in loader) == 4
    assert sum(1 for _ in loader) == 4


def test_large_prefetch(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(
        ds,
        batch_size=1024,
        num_steps=9,
        columns=["f1", "f2", "label"],
        num_workers=4,
        prefetch_factor=8,
    )
    assert sum(1 for _ in loader) == 9


def test_torch_tensor(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=256, num_steps=4, columns=["f1", "label"])

    for batch in loader:
        x = torch.from_numpy(batch["f1"].copy())
        y = torch.from_numpy(batch["label"].copy())

        assert x.dtype == torch.float32
        assert y.dtype == torch.int32


def test_torch_from_numpy_zero_copy(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=256, num_steps=1, columns=["f1"])

    for batch in loader:
        with pytest.warns(UserWarning, match="The given NumPy array is not writable") as record:
            _ = torch.from_numpy(batch["f1"])

        assert len(record) == 1

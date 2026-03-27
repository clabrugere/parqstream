import numpy as np
import pytest
import torch
from parqstream import DataLoader, Dataset


def test_basic_iteration(parquet_path):
    ds = Dataset([parquet_path], columns=["f1", "label"])
    loader = DataLoader(
        ds,
        batch_size=256,
        num_steps=4,
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
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3)

    for batch in loader:
        assert len(batch["f1"]) == 512


def test_all_columns_default(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=128, num_steps=4)

    for batch in loader:
        assert set(batch.keys()) == set(ds.columns)


def test_multi_file(two_parquet_paths):
    ds = Dataset(two_parquet_paths)
    loader = DataLoader(ds, batch_size=256, num_steps=4)

    assert sum(1 for _ in loader) == 4


def test_multiple_iterations(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(ds, batch_size=256, num_steps=4)

    assert sum(1 for _ in loader) == 4
    assert sum(1 for _ in loader) == 4


def test_large_prefetch(parquet_path):
    ds = Dataset([parquet_path])
    loader = DataLoader(
        ds,
        batch_size=1024,
        num_steps=9,
        num_workers=4,
        prefetch_factor=8,
    )
    assert sum(1 for _ in loader) == 9


def test_torch_tensor(parquet_path):
    ds = Dataset([parquet_path], columns=["f1", "label"])
    loader = DataLoader(ds, batch_size=256, num_steps=4)

    for batch in loader:
        x = torch.from_numpy(batch["f1"].copy())
        y = torch.from_numpy(batch["label"].copy())

        assert x.dtype == torch.float32
        assert y.dtype == torch.int32


def test_torch_from_numpy_zero_copy(parquet_path):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=256, num_steps=1)

    for batch in loader:
        with pytest.warns(UserWarning, match="The given NumPy array is not writable") as record:
            _ = torch.from_numpy(batch["f1"])

        assert len(record) == 1


def test_sequential_order(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 1_000
    num_steps = 10

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_steps=num_steps,
        shuffle=False,
        num_workers=1,
    )

    all_ids = np.concatenate([batch["id"] for batch in loader])
    assert np.array_equal(all_ids, np.arange(10_000, dtype=np.int64))


def test_sequential_wraps_around(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 1_000
    num_steps = 15

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_steps=num_steps,
        shuffle=False,
        num_workers=1,
    )

    all_ids = np.concatenate([batch["id"] for batch in loader])
    expected = np.arange(15_000, dtype=np.int64) % 10_000
    assert np.array_equal(all_ids, expected)


def test_shuffle_ids_in_bounds(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])

    loader = DataLoader(
        ds,
        batch_size=1_000,
        num_steps=10,
        shuffle=True,
    )

    all_ids = np.concatenate([batch["id"] for batch in loader])
    assert all_ids.min() >= 0
    assert all_ids.max() < 10_000
    assert len(all_ids) == 10_000

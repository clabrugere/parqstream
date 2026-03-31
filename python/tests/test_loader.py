import numpy as np
import pytest
import torch
from parqstream import DataLoader, Dataset


def test_basic_iteration(parquet_path):
    ds = Dataset([parquet_path], columns=["f1", "label"])
    loader = DataLoader(ds, batch_size=256, num_steps=4, num_workers=2)

    for batch in loader:
        assert set(batch.keys()) == {"f1", "label"}
        assert batch["f1"].dtype == np.float32
        assert batch["label"].dtype == np.int32
        assert len(batch["f1"]) > 0


def test_infinite_iteration(parquet_path):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=None)

    for i, batch in enumerate(loader):
        assert len(batch["f1"]) == 512
        if i >= 5:
            break


def test_dataloader_len(parquet_path):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3)

    assert sum(1 for _ in loader) == 3


def test_dataloader_len_with_buffer(parquet_path):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3, buffer_size=1024)

    assert sum(1 for _ in loader) == 3


def test_batch_size(parquet_path):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3)

    for batch in loader:
        assert len(batch["f1"]) == 512


def test_batch_size_with_buffer(parquet_path):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3, buffer_size=1024)

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
    loader = DataLoader(ds, batch_size=1024, num_steps=9, num_workers=4, prefetch_factor=8)
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

    loader = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=False, num_workers=1)

    all_ids = np.concatenate([batch["id"] for batch in loader])
    assert np.array_equal(all_ids, np.arange(10_000, dtype=np.int64))


def test_sequential_wraps_around(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 1_000
    num_steps = 15

    loader = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=False, num_workers=1)

    all_ids = np.concatenate([batch["id"] for batch in loader])
    expected = np.arange(15_000, dtype=np.int64) % 10_000
    assert np.array_equal(all_ids, expected)


def test_shuffle_ids_in_bounds(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])

    loader = DataLoader(ds, batch_size=1_000, num_steps=10, shuffle=True)

    all_ids = np.concatenate([batch["id"] for batch in loader])
    assert all_ids.min() >= 0
    assert all_ids.max() < 10_000
    assert len(all_ids) == 10_000


def test_buffer_explicit_size(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader = DataLoader(ds, batch_size=500, num_steps=20, shuffle=True, buffer_size=2_000)

    all_ids = np.concatenate([batch["id"] for batch in loader])
    assert all_ids.min() >= 0
    assert all_ids.max() < 10_000
    assert len(all_ids) == 10_000


def test_no_row_skip_batch_not_divisible_by_row_group(parquet_path):
    # batch_size=300 does not divide row_group_size=1_000; the 100-row remainder of each row group must be stitched into the next batch, not dropped.
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 300
    num_steps = 10_000 // batch_size  # 33 full batches = 9_900 rows

    loader = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=False, num_workers=1)
    all_ids = np.concatenate([batch["id"] for batch in loader])

    assert len(all_ids) == num_steps * batch_size
    assert np.array_equal(all_ids, np.arange(num_steps * batch_size, dtype=np.int64))


def test_no_row_skip_buffer_not_divisible_by_row_group(parquet_path):
    # buffer_size=1_500 straddles row-group boundaries (1.5 row groups per fill). the 500-row tail of each buffer must be prepended to the next fill, not dropped.
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 500
    buffer_size = 1_500
    num_steps = 10_000 // batch_size  # 20 full batches = 10_000 rows

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_steps=num_steps,
        shuffle=False,
        num_workers=1,
        buffer_size=buffer_size,
    )
    all_ids = np.concatenate([batch["id"] for batch in loader])

    assert len(all_ids) == 10_000
    assert np.array_equal(all_ids, np.arange(10_000, dtype=np.int64))


def test_no_row_skip_batch_not_divisible_by_buffer(parquet_path):
    # batch_size=300 does not divide buffer_size=1_000; the 100-row remainder left in the buffer after 3 full batches must carry over to the next refill.
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 300
    buffer_size = 1_000
    num_steps = 10_000 // batch_size  # 33 full batches = 9_900 rows

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_steps=num_steps,
        shuffle=False,
        num_workers=1,
        buffer_size=buffer_size,
    )
    all_ids = np.concatenate([batch["id"] for batch in loader])

    assert len(all_ids) == num_steps * batch_size
    assert np.array_equal(all_ids, np.arange(num_steps * batch_size, dtype=np.int64))


def test_no_row_skip_wrap_around_non_divisible(parquet_path):
    # Wrap past one full epoch with non-divisible sizes; rows must be contiguous across the dataset boundary with no gaps or duplicates.
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 300
    buffer_size = 1_500
    num_steps = 40  # > one epoch (33.3 steps), forces wrap-around

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_steps=num_steps,
        shuffle=False,
        num_workers=1,
        buffer_size=buffer_size,
    )
    all_ids = np.concatenate([batch["id"] for batch in loader])
    expected = np.arange(num_steps * batch_size, dtype=np.int64) % 10_000

    assert len(all_ids) == num_steps * batch_size
    assert np.array_equal(all_ids, expected)

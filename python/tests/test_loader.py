import numpy as np
import pyarrow as pa
import pytest
import torch
from parqstream import DataLoader, Dataset, _col_to_numpy


def test_zero_copy_column():
    arr = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    col = arr  # or a Column wrapping it
    result = _col_to_numpy(col)
    assert np.shares_memory(result, arr.buffers()[1])  # same underlying buffer


def test_copy_fallback_column():
    arr = pa.array([1, None, 3], type=pa.int64())
    col = arr
    with pytest.warns(UserWarning, match="falling back"):  # or check logger
        result = _col_to_numpy(col)
    assert not np.shares_memory(result, arr.buffers()[1])
    assert np.isnan(result[1])  # null becomes NaN


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
    num_batches_in_one_epoch = len(ds) // 512 + 1

    for i, batch in enumerate(loader):
        if i > num_batches_in_one_epoch:
            assert len(batch["f1"]) == 512
            break


@pytest.mark.parametrize("buffer_size", [None, 1024])
def test_dataloader_len(parquet_path, buffer_size):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3, buffer_size=buffer_size)

    assert len(loader) == 3
    assert sum(1 for _ in loader) == 3


def test_len_infinite(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader = DataLoader(ds, batch_size=512, num_steps=None)

    with pytest.raises(ValueError, match="length is undefined for infinite DataLoader"):
        _ = len(loader)


@pytest.mark.parametrize("buffer_size", [None, 1024])
def test_batch_size(parquet_path, buffer_size):
    ds = Dataset([parquet_path], columns=["f1"])
    loader = DataLoader(ds, batch_size=512, num_steps=3, buffer_size=buffer_size)

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


@pytest.mark.parametrize("num_steps", [10, 15])
def test_sequential_order(parquet_path, num_steps):
    ds = Dataset([parquet_path], columns=["id"])
    batch_size = 1_000

    loader = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=False, num_workers=1)
    all_ids = np.concatenate([batch["id"] for batch in loader])
    expected = np.arange(num_steps * batch_size, dtype=np.int64) % 10_000

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


def test_seeded_shuffle_is_reproducible(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=512, num_steps=10, shuffle=True, seed=42)

    ids_a = [batch["id"].tolist() for batch in DataLoader(ds, **kwargs)]
    ids_b = [batch["id"].tolist() for batch in DataLoader(ds, **kwargs)]

    assert ids_a == ids_b


def test_seeded_shuffle_differs_each_epoch(parquet_path):
    # Each __iter__ call should produce a different order (different epoch seed)
    # otherwise the model sees the same batch sequence every epoch.
    ds = Dataset([parquet_path], columns=["id"])
    loader = DataLoader(ds, batch_size=512, num_steps=10, shuffle=True, seed=42)

    ids_a = [batch["id"].tolist() for batch in loader]
    ids_b = [batch["id"].tolist() for batch in loader]

    assert ids_a != ids_b


def test_different_seeds_differ(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=512, num_steps=5, shuffle=True)

    ids_42 = [batch["id"].tolist() for batch in DataLoader(ds, **kwargs, seed=42)]
    ids_99 = [batch["id"].tolist() for batch in DataLoader(ds, **kwargs, seed=99)]

    assert ids_42 != ids_99


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


def test_collate_fn_returns_record_batch(parquet_path):
    ds = Dataset([parquet_path], columns=["f1", "label"])
    loader = DataLoader(
        ds,
        batch_size=256,
        num_steps=2,
        collate_fn=lambda b: pa.RecordBatchReader.from_stream(b).read_next_batch(),
    )
    for rb in loader:
        assert isinstance(rb, pa.RecordBatch)
        assert len(rb) == 256


def test_collate_fn_torch(parquet_path):
    def collate_fn(batch):
        return {col.name: torch.from_numpy(pa.array(col).to_numpy(zero_copy_only=False)) for col in batch.columns()}

    ds = Dataset([parquet_path], columns=["f1", "label"])
    loader = DataLoader(ds, batch_size=256, num_steps=4, collate_fn=collate_fn)

    for batch in loader:
        assert isinstance(batch["f1"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_from_checkpoint(parquet_path, shuffle):
    seed = 42
    ds = Dataset([parquet_path], columns=["id"])

    # complete uninterrupted run used as reference
    ref = DataLoader(ds, batch_size=512, num_steps=10, shuffle=shuffle, seed=seed)
    reference = np.concatenate([b["id"] for b in ref])

    # interrupted run: 3 steps, checkpoint, resume
    loader = DataLoader(ds, batch_size=512, num_steps=10, shuffle=shuffle, seed=seed)
    it = iter(loader)
    first = np.concatenate([next(it)["id"] for _ in range(3)])

    new_loader = DataLoader(ds, batch_size=512, num_steps=10, shuffle=shuffle, seed=seed)
    new_loader.load_state_dict(loader.state_dict())
    second = np.concatenate([b["id"] for b in new_loader])

    assert np.array_equal(np.concatenate([first, second]), reference)


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_from_checkpoint_multiple_epoch(parquet_path, shuffle):
    num_steps = 40  # > one epoch (≈19.5 steps), forces internal wrap in chunk_feeder
    batch_size = 512
    seed = 42
    ds = Dataset([parquet_path], columns=["id"])

    ref = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=shuffle, seed=seed)
    reference = np.concatenate([b["id"] for b in ref])

    loader = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=shuffle, seed=seed)
    it = iter(loader)
    first = np.concatenate([next(it)["id"] for _ in range(22)])  # past 1 full epoch

    new_loader = DataLoader(ds, batch_size=batch_size, num_steps=num_steps, shuffle=shuffle, seed=seed)
    new_loader.load_state_dict(loader.state_dict())
    second = np.concatenate([b["id"] for b in new_loader])

    assert np.array_equal(np.concatenate([first, second]), reference)


def test_resume_different_dataset(parquet_path):
    ds1 = Dataset([parquet_path], columns=["id"])
    ds2 = Dataset([parquet_path], columns=["f1"])

    loader = DataLoader(ds1, batch_size=512, num_steps=10)
    _ = next(iter(loader))
    state = loader.state_dict()

    new_loader = DataLoader(ds2, batch_size=512, num_steps=10)
    with pytest.raises(ValueError, match="dataset identifier mismatch"):
        new_loader.load_state_dict(state)


def test_state_dict_before_iter_raises(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader = DataLoader(ds, batch_size=512, num_steps=10)

    with pytest.raises(RuntimeError, match="iter"):
        loader.state_dict()


def test_resume_after_full_consumption(parquet_path):
    # Checkpointing after all batches are exhausted records steps_remaining=0.
    # Loading that checkpoint and iterating should yield no batches.
    ds = Dataset([parquet_path], columns=["id"])

    loader = DataLoader(ds, batch_size=128, num_steps=5)
    _ = list(loader)  # consume all 5 batches

    state = loader.state_dict()

    new_loader = DataLoader(ds, batch_size=128, num_steps=5)
    new_loader.load_state_dict(state)
    assert len(list(new_loader)) == 0

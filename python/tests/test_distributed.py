import numpy as np
import pytest
from parqstream import DataLoader, Dataset


# --- Basic coverage and disjointness ---


@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed_disjoint_full_coverage(parquet_path, shuffle):
    ds = Dataset([parquet_path], columns=["id"])
    num_steps = 10_000 // 2 // 500  # one epoch per rank: 10 steps
    kwargs = dict(batch_size=500, num_steps=num_steps, shuffle=shuffle, seed=42)

    ids0 = np.sort(np.concatenate([b["id"] for b in DataLoader(ds, **kwargs, rank=0, world_size=2)]))
    ids1 = np.sort(np.concatenate([b["id"] for b in DataLoader(ds, **kwargs, rank=1, world_size=2)]))

    assert np.array_equal(np.sort(np.concatenate([ids0, ids1])), np.arange(10_000, dtype=np.int64))
    assert len(np.intersect1d(ids0, ids1)) == 0


def test_distributed_ranks_yield_different_ids(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=500, num_steps=10, shuffle=False)

    ids0 = np.concatenate([b["id"] for b in DataLoader(ds, **kwargs, rank=0, world_size=2)])
    ids1 = np.concatenate([b["id"] for b in DataLoader(ds, **kwargs, rank=1, world_size=2)])

    assert not np.array_equal(ids0, ids1)


# --- Reproducibility and epoch variation ---


def test_distributed_seeded_is_reproducible(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=500, num_steps=10, shuffle=True, seed=42, rank=0, world_size=2)

    ids_a = [b["id"].tolist() for b in DataLoader(ds, **kwargs)]
    ids_b = [b["id"].tolist() for b in DataLoader(ds, **kwargs)]

    assert ids_a == ids_b


def test_distributed_shuffle_differs_each_epoch(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader = DataLoader(ds, batch_size=500, num_steps=10, shuffle=True, seed=42, rank=0, world_size=2)

    ids_a = [b["id"].tolist() for b in loader]
    ids_b = [b["id"].tolist() for b in loader]

    assert ids_a != ids_b


# --- Checkpoint resume ---


@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed_resume_from_checkpoint(parquet_path, shuffle):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=500, num_steps=10, shuffle=shuffle, seed=42, rank=1, world_size=2)

    reference = np.concatenate([b["id"] for b in DataLoader(ds, **kwargs)])

    loader = DataLoader(ds, **kwargs)
    it = iter(loader)
    first = np.concatenate([next(it)["id"] for _ in range(3)])

    resumed = DataLoader(ds, **kwargs)
    resumed.load_state_dict(loader.state_dict())
    second = np.concatenate([b["id"] for b in resumed])

    assert np.array_equal(np.concatenate([first, second]), reference)


@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed_resume_across_epoch_boundary(parquet_path, shuffle):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=500, num_steps=25, shuffle=shuffle, seed=42, rank=0, world_size=2)

    reference = np.concatenate([b["id"] for b in DataLoader(ds, **kwargs)])

    loader = DataLoader(ds, **kwargs)
    it = iter(loader)
    first = np.concatenate([next(it)["id"] for _ in range(12)])  # past one full rank-epoch (10 steps)

    resumed = DataLoader(ds, **kwargs)
    resumed.load_state_dict(loader.state_dict())
    second = np.concatenate([b["id"] for b in resumed])

    assert np.array_equal(np.concatenate([first, second]), reference)


# --- Checkpoint validation ---


def test_distributed_checkpoint_rank_mismatch_raises(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader0 = DataLoader(ds, batch_size=500, num_steps=10, rank=0, world_size=2)
    _ = next(iter(loader0))

    loader1 = DataLoader(ds, batch_size=500, num_steps=10, rank=1, world_size=2)
    with pytest.raises(ValueError, match="distributed config mismatch"):
        loader1.load_state_dict(loader0.state_dict())


def test_distributed_checkpoint_world_size_mismatch_raises(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader2 = DataLoader(ds, batch_size=500, num_steps=10, rank=0, world_size=2)
    _ = next(iter(loader2))

    loader4 = DataLoader(ds, batch_size=500, num_steps=10, rank=0, world_size=4)
    with pytest.raises(ValueError, match="distributed config mismatch"):
        loader4.load_state_dict(loader2.state_dict())


# --- Constructor validation ---


def test_invalid_rank_raises(parquet_path):
    ds = Dataset([parquet_path])
    with pytest.raises(ValueError):
        DataLoader(ds, batch_size=500, rank=2, world_size=2)


def test_invalid_world_size_zero_raises(parquet_path):
    ds = Dataset([parquet_path])
    with pytest.raises(ValueError):
        DataLoader(ds, batch_size=500, world_size=0)


def test_world_size_exceeds_row_groups_raises(parquet_path):
    # parquet_path has 10 row groups; world_size=11 means rank 10 gets 0 groups
    ds = Dataset([parquet_path])
    with pytest.raises(ValueError, match="row group"):
        DataLoader(ds, batch_size=500, rank=0, world_size=11)

import numpy as np
from parqstream import DataLoader, Dataset


# --- Shuffle correctness ---


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


# --- Reproducibility ---


def test_seeded_shuffle_is_reproducible(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    kwargs = dict(batch_size=512, num_steps=10, shuffle=True, seed=42)

    ids_a = [batch["id"].tolist() for batch in DataLoader(ds, **kwargs)]
    ids_b = [batch["id"].tolist() for batch in DataLoader(ds, **kwargs)]

    assert ids_a == ids_b


def test_seeded_shuffle_differs_each_epoch(parquet_path):
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

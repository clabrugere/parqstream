import numpy as np
import pytest
from parqstream import DataLoader, Dataset


# --- Resume correctness ---


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_from_checkpoint(parquet_path, shuffle):
    seed = 42
    ds = Dataset([parquet_path], columns=["id"])

    ref = DataLoader(ds, batch_size=512, num_steps=10, shuffle=shuffle, seed=seed)
    reference = np.concatenate([b["id"] for b in ref])

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


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_from_checkpoint_twice(parquet_path, shuffle):
    args = {
        "dataset": Dataset([parquet_path], columns=["id"]),
        "batch_size": 512,
        "num_steps": 10,
        "shuffle": shuffle,
        "seed": 42,
    }

    ref = DataLoader(**args)
    reference = np.concatenate([b["id"] for b in ref])

    loader = DataLoader(**args)
    it = iter(loader)
    first = np.concatenate([next(it)["id"] for _ in range(3)])

    first_resume = DataLoader(**args)
    first_resume.load_state_dict(loader.state_dict())
    it = iter(first_resume)
    second = np.concatenate([next(it)["id"] for _ in range(3)])

    second_resume = DataLoader(**args)
    second_resume.load_state_dict(first_resume.state_dict())
    third = np.concatenate([b["id"] for b in second_resume])

    assert np.array_equal(np.concatenate([first, second, third]), reference)


def test_resume_after_full_consumption(parquet_path):
    # Checkpointing after all batches are exhausted records steps_remaining=0.
    # Loading that checkpoint and iterating should yield no batches.
    ds = Dataset([parquet_path], columns=["id"])

    loader = DataLoader(ds, batch_size=128, num_steps=5)
    _ = list(loader)

    state = loader.state_dict()

    new_loader = DataLoader(ds, batch_size=128, num_steps=5)
    new_loader.load_state_dict(state)
    assert len(list(new_loader)) == 0


# --- Checkpoint validation ---


def test_resume_different_dataset(parquet_path):
    ds1 = Dataset([parquet_path], columns=["id"])
    ds2 = Dataset([parquet_path], columns=["f1"])

    loader = DataLoader(ds1, batch_size=512, num_steps=10)
    _ = next(iter(loader))
    state = loader.state_dict()

    new_loader = DataLoader(ds2, batch_size=512, num_steps=10)
    with pytest.raises(ValueError, match="dataset mismatch"):
        new_loader.load_state_dict(state)


def test_state_dict_before_iter_raises(parquet_path):
    ds = Dataset([parquet_path], columns=["id"])
    loader = DataLoader(ds, batch_size=512, num_steps=10)

    with pytest.raises(RuntimeError, match="iter"):
        loader.state_dict()

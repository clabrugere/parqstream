import pytest
from parqstream import Dataset


def test_open_single_file(parquet_path):
    ds = Dataset([parquet_path])
    assert len(ds) == 10_000
    assert ds.num_files == 1
    assert ds.num_row_groups == 10  # 10_000 rows / 1_000 per group


def test_column_names(parquet_path):
    ds = Dataset([parquet_path])
    assert set(ds.column_names) == {"label", "f1", "f2", "weight", "id"}


def test_open_multiple_files(two_parquet_paths):
    ds = Dataset(two_parquet_paths)
    assert len(ds) == 10_000  # 2 × 5_000
    assert ds.num_files == 2
    assert ds.num_row_groups == 10  # 2 × (5_000 / 1_000)


def test_empty_paths_raises():
    with pytest.raises(Exception):
        Dataset([])


def test_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        Dataset([str(tmp_path / "nonexistent.parquet")])


def test_repr(parquet_path):
    ds = Dataset([parquet_path])
    r = repr(ds)
    assert "Dataset" in r
    assert "10000" in r

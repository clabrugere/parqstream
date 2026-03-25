import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

N_ROWS = 10_000
ROW_GROUP_SIZE = 1_000  # 10 row groups


@pytest.fixture(scope="session")
def parquet_path(tmp_path_factory):
    """Create a single parquet file with numeric + label columns."""
    rng = np.random.default_rng(42)
    n = N_ROWS

    table = pa.table(
        {
            "label": pa.array(rng.integers(0, 2, n, dtype=np.int32)),
            "f1": pa.array(rng.standard_normal(n).astype(np.float32)),
            "f2": pa.array(rng.standard_normal(n).astype(np.float32)),
            "weight": pa.array(rng.uniform(0, 1, n).astype(np.float32)),
            "id": pa.array(np.arange(n, dtype=np.int64)),
        }
    )

    path = str(tmp_path_factory.mktemp("data") / "train.parquet")
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
    return path


@pytest.fixture(scope="session")
def two_parquet_paths(tmp_path_factory):
    """Two parquet files with the same schema."""
    rng = np.random.default_rng(0)
    tmpdir = tmp_path_factory.mktemp("data2")
    paths = []

    for i in range(2):
        n = 5_000
        table = pa.table(
            {
                "label": pa.array(rng.integers(0, 2, n, dtype=np.int32)),
                "f1": pa.array(rng.standard_normal(n).astype(np.float32)),
            }
        )
        path = str(tmpdir / f"shard_{i}.parquet")
        pq.write_table(table, path, row_group_size=1_000)
        paths.append(path)

    return paths

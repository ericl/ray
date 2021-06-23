import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import builtins
import ray

from ray.experimental.data import ArrowDataset, ArrowBlock
from ray.tests.conftest import *  # noqa


def test_basic(ray_start_regular_shared):
    ds = ray.experimental.data.range(5)
    assert ds.map(lambda x: x + 1).take() == [1, 2, 3, 4, 5]


def test_parquet(ray_start_regular_shared, tmp_path):
    df1 = pd.DataFrame({"one": [1, 2, 3], "two": ["a", "b", "c"]})
    table = pa.Table.from_pandas(df1)
    pq.write_table(table, os.path.join(tmp_path, "test1.parquet"))
    df2 = pd.DataFrame({"one": [4, 5, 6], "two": ["e", "f", "g"]})
    table = pa.Table.from_pandas(df2)
    pq.write_table(table, os.path.join(tmp_path, "test2.parquet"))

    ds = ray.experimental.data.read_parquet(tmp_path)
    assert sorted(ds.take()) == [[4, 'e'], [4, 'e'], [5, 'f'], [5, 'f'],
                                 [6, 'g'], [6, 'g']]


def range_arrow(n: int, num_blocks: int = 200) -> "ArrowDataset":
    block_size = max(1, n // num_blocks)
    blocks: List[BlockRef] = []
    i = 0

    @ray.remote
    def gen_block(start: int, count: int) -> "ArrowBlock":
        return {"a": list(builtins.range(start, start + count))}

    while i < n:
        blocks.append(gen_block.remote(block_size * i, min(block_size, n - i)))
        i += block_size

    return ArrowDataset(blocks, ArrowBlock)


def test_pyarrow(ray_start_regular_shared):
    ds = range_arrow(5)
    assert ds.map(lambda x: {"b": x["a"] + 2}).take() == \
        [{"b": 2}, {"b": 3}, {"b": 4}, {"b": 5}, {"b": 6}]
    assert ds.map(lambda x: {"b": x["a"] + 2}).filter(lambda x: x["b"] % 2 == 0).take() == \
        [{"b": 2}, {"b": 4}, {"b": 6}]
    assert ds.filter(lambda x: x["a"] == 0).flat_map(lambda x: [{"b": x["a"] + 2}, {"b": x["a"] + 20}]).take() == \
        [{"b": 2}, {"b": 20}]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))

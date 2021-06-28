from typing import List, Any, Callable, Iterable, Iterator, Generic, TypeVar, \
    Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow
    import pandas
    import modin
    import dask
    import pyspark
    import ray.util.sgd

import itertools
import ray
from ray.experimental.data.impl.compute import get_compute
from ray.experimental.data.impl.shuffle import simple_shuffle
from ray.experimental.data.impl.block import ObjectRef, Block
from ray.experimental.data.impl.arrow_block import DelegatingArrowBlockBuilder
from ray.experimental import get_object_locations

T = TypeVar("T")
U = TypeVar("U")
BatchType = Union["pandas.DataFrame", "pyarrow.Table"]


class Dataset(Generic[T]):
    """Implements a distributed Arrow dataset.

    Datasets are implemented as a list of ``ObjectRef[Block[T]]``. The block
    also determines the unit of parallelism. The default block type is the
    ``ArrowBlock``, which is backed by a ``pyarrow.Table`` object. Other
    Python objects are represented with ``ListBlock`` (a plain Python list).

    Since Datasets are just lists of Ray object refs, they can be passed
    between Ray tasks and actors just like any other object. Datasets support
    conversion to/from several more featureful dataframe libraries
    (e.g., Spark, Dask), and are also compatible with TensorFlow / PyTorch.

    Dataset supports parallel transformations such as .map(), .map_batches(),
    and simple repartition, but currently not aggregations and joins.
    """

    def __init__(self, blocks: Iterable[ObjectRef[Block[T]]]):
        self._blocks: Iterable[ObjectRef[Block[T]]] = blocks

    def map(self, fn: Callable[[T], U], compute="tasks",
            **ray_remote_args) -> "Dataset[U]":
        """Apply the given function to each record of this dataset.

        This is a blocking operation. Note that mapping individual records
        can be quite slow. Consider using `.map_batches()` for performance.

        Examples:
            # Transform python objects.
            >>> ds.map(lambda x: x * 2)

            # Transform Arrow records.
            >>> ds.map(lambda record: {"v2": record["value"] * 2})

        Time complexity: O(dataset size / parallelism)

        Args:
            fn: The function to apply to each record.
            compute: The compute strategy, either "tasks" to use Ray tasks,
                or "actors" to use an autoscaling Ray actor pool.
            ray_remote_args: Additional resource requirements to request from
                ray (e.g., num_gpus=1 to request GPUs for the map tasks).
        """

        def transform(block: Block[T]) -> Block[U]:
            builder = DelegatingArrowBlockBuilder()
            for row in block.iter_rows():
                builder.add(fn(row))
            return builder.build()

        compute = get_compute(compute)

        return Dataset(compute.apply(transform, ray_remote_args, self._blocks))

    def map_batches(self,
                    fn: Callable[[BatchType], BatchType],
                    batch_size: int = None,
                    compute: str = "tasks",
                    batch_format: str = "pandas",
                    **ray_remote_args) -> "Dataset[Any]":
        """Apply the given function to batches of records of this dataset.

        This is a blocking operation.

        Examples:
            # Transform batches in parallel.
            >>> ds.map_batches(lambda batch: [v * 2 for v in batch])

            # Transform batches in parallel on GPUs.
            >>> ds.map_batches(
            ...    batch_infer_fn,
            ...    batch_size=256, compute="actors", num_gpus=1)

        Time complexity: O(dataset size / parallelism)

        Args:
            fn: The function to apply to each record batch.
            batch_size: Request a specific batch size, or leave unspecified
                to use entire blocks as batches.
            compute: The compute strategy, either "tasks" to use Ray tasks,
                or "actors" to use an autoscaling Ray actor pool.
            batch_format: Specify "pandas" to select ``pandas.DataFrame`` as
                the batch format, or "pyarrow" to select ``pyarrow.Table``.
            ray_remote_args: Additional resource requirements to request from
                ray (e.g., num_gpus=1 to request GPUs for the map tasks).
        """
        raise NotImplementedError  # P0

    def flat_map(self,
                 fn: Callable[[T], Iterable[U]],
                 compute="tasks",
                 **ray_remote_args) -> "Dataset[U]":
        """Apply the given function to each record and then flatten results.

        This is a blocking operation. Consider using ``.map_batches()`` for
        better performance (the batch size can be altered in map_batches).

        Examples:
            >>> ds.flat_map(lambda x: [x, x ** 2, x ** 3])

        Time complexity: O(dataset size / parallelism)

        Args:
            fn: The function to apply to each record.
            compute: The compute strategy, either "tasks" to use Ray tasks,
                or "actors" to use an autoscaling Ray actor pool.
            ray_remote_args: Additional resource requirements to request from
                ray (e.g., num_gpus=1 to request GPUs for the map tasks).
        """

        def transform(block: Block[T]) -> Block[U]:
            builder = DelegatingArrowBlockBuilder()
            for row in block.iter_rows():
                for r2 in fn(row):
                    builder.add(r2)
            return builder.build()

        compute = get_compute(compute)

        return Dataset(compute.apply(transform, ray_remote_args, self._blocks))

    def filter(self,
               fn: Callable[[T], bool],
               compute="tasks",
               **ray_remote_args) -> "Dataset[T]":
        """Filter out records that do not satisfy the given predicate.

        This is a blocking operation. Consider using ``.map_batches()`` for
        better performance (you can implement filter by dropping records).

        Examples:
            >>> ds.flat_map(lambda x: x % 2 == 0)

        Time complexity: O(dataset size / parallelism)

        Args:
            fn: The predicate function to apply to each record.
            compute: The compute strategy, either "tasks" to use Ray tasks,
                or "actors" to use an autoscaling Ray actor pool.
            ray_remote_args: Additional resource requirements to request from
                ray (e.g., num_gpus=1 to request GPUs for the map tasks).
        """

        def transform(block: Block[T]) -> Block[T]:
            builder = block.builder()
            for row in block.iter_rows():
                if fn(row):
                    builder.add(row)
            return builder.build()

        compute = get_compute(compute)

        return Dataset(compute.apply(transform, ray_remote_args, self._blocks))

    def repartition(self, num_blocks: int) -> "Dataset[T]":
        """Repartition the dataset into exactly this number of blocks.

        This is a blocking operation.

        Examples:
            # Set the number of output partitions to write to disk.
            >>> ds.repartition(100).write_parquet(...)

        Time complexity: O(dataset size / parallelism)

        Args:
            num_blocks: The number of blocks.

        Returns:
            The repartitioned dataset.
        """

        new_blocks = simple_shuffle(self._blocks, num_blocks)
        return Dataset(new_blocks)

    def split(self, n: int,
              locality_hints: List[Any] = None) -> List["Dataset[T]"]:
        """Split the dataset into ``n`` disjoint pieces.

        This returns a list of sub-datasets that can be passed to Ray tasks
        and actors and used to read the dataset records in parallel.

        Examples:
            >>> # Split up a dataset to process over `n` worker actors.
            >>> shards = ds.split(len(workers), locality_hints=workers)
            >>> for shard, worker in zip(shards, workers):
            ...     worker.consume.remote(shard)

        Time complexity: O(1)

        Args:
            n: Number of child datasets to return.
            locality_hints: A list of Ray actor handles of size ``n``. The
                system will try to co-locate the blocks of the ith dataset
                with the ith actor to maximize data locality.

        Returns:
            A list of ``n`` disjoint dataset splits.
        """
        assert n > 0 and n <= len(self._blocks)
        assert locality_hints is None or len(locality_hints) == n

        # Split self._blocks into n chunks. the first {num_large_chunks} chunks
        # have {small_chunk_size + 1} blocks, and remaining
        # chunks have {small_chunk_size} blocks.
        small_chunk_size = len(self._blocks) // n
        num_large_chunks = len(self._blocks) - small_chunk_size * n

        datasets = []
        block_refs = list(self._blocks)

        if locality_hints is None:
            cur = 0
            for i in range(n):
                end = cur + small_chunk_size
                if i < num_large_chunks:
                    end = end + 1
                datasets.append(Dataset(block_refs[cur:end]))
                cur = end
            return datasets

        # If the locality_hints is set, we use a greedy algorithm
        # to co-locate the blocks with the actors based on block
        # and actor's node_id:
        #
        # 1. We get the node_id where the block resides in and store it in
        # a map {node_id: [blocks]}. Note that each block could
        # be located on multiple nodes. For simplicity, we use the first
        # node_id so that there is one to one mapping from node_id to block
        # in the map.
        #
        # 2. We do two round of allocations. In the first round, we allocate
        # blocks for each actor based by sustract the blocks from the map
        # we build from previous step.
        #
        # 3. Some actors might have more blocks than needed while the
        # others don't have enough blocks. We do second round of allocation
        # for the reminder blocks so that every actor have sufficient blocks.

        # wait might affect the throughput due to stragglers?
        ray.wait(block_refs)

        # build node_id to block_refs map
        block_ref_locations = get_object_locations(block_refs)
        block_refs_by_node_id = {}
        for block_ref in block_refs:
            node_ids = block_ref_locations.get(block_ref, {}).get(
                "node_ids", [])
            node_id = node_ids[0] if node_ids else None
            if node_id not in block_refs_by_node_id:
                block_refs_by_node_id[node_id] = []
            block_refs_by_node_id[node_id].append(block_ref)

        # find the node_ids for each actors and the num of blocks
        # needed for each actor
        actors_state = ray.state.actors()
        actor_node_ids = [
            actors_state.get(actor._actor_id.hex(), {}).get("Address",
                                                            {}).get("NodeID")
            for actor in locality_hints
        ]
        num_blocks_by_actor = [small_chunk_size + 1] * num_large_chunks + [
            small_chunk_size
        ] * (n - num_large_chunks)

        # first round of allocation by substracting the blocks from
        # block_refs_by_node_id map based on each actor's node_id.
        block_refs_by_actor = []
        for i in range(n):
            node_id = actor_node_ids[i]
            block_refs = block_refs_by_node_id.get(node_id, [])
            block_refs_by_actor.append(block_refs[:num_blocks_by_actor[i]])
            del block_refs[:num_blocks_by_actor[i]]
            num_blocks_by_actor[i] -= len(block_refs_by_actor[i])

        # second round of allocation for remaining blocks
        remaining_block_refs = list(
            itertools.chain.from_iterable(block_refs_by_node_id.values()))

        for i in range(n):
            block_refs_by_actor[i].extend(
                remaining_block_refs[:num_blocks_by_actor[i]])
            del remaining_block_refs[:num_blocks_by_actor[i]]

        return [Dataset(block_refs) for block_refs in block_refs_by_actor]

    def sort(self,
             key: Union[None, str, List[str], Callable[[T], Any]],
             descending: bool = False) -> "Dataset[T]":
        """Sort the dataset by the specified key columns or key function.

        This is a blocking operation.

        Examples:
            # Sort using the entire record as the key.
            >>> ds.sort()

            # Sort by a single column.
            >>> ds.sort("field1")

            # Sort by multiple columns.
            >>> ds.sort(["field1", "field2"])

            # Sort by a key function.
            >>> ds.sort(lambda record: record["field1"] % 100)

        Time complexity: O(dataset size / parallelism)

        Args:
            key: Either a single Arrow column name, a list of Arrow column
                names, a function that returns a sortable key given each
                record as an input, or None to sort by the entire record.
            descending: Whether to sort in descending order.

        Returns:
            The sorted dataset.
        """
        raise NotImplementedError  # P2

    def limit(self, limit: int) -> "Dataset[T]":
        """Limit the dataset to the first number of records specified.

        Examples:
            >>> ds.limit(100).map(lambda x: x * 2).take()

        Time complexity: O(limit specified)

        Args:
            limit: The size of the dataset to truncate to.

        Returns:
            The truncated dataset.
        """
        raise NotImplementedError  # P1

    def take(self, limit: int = 20) -> List[T]:
        """Take up to the given number of records from the dataset.

        Time complexity: O(limit specified)

        Args:
            limit: The max number of records to return.

        Returns:
            A list of up to ``limit`` records from the dataset.
        """
        output = []
        for row in self.iter_rows():
            output.append(row)
            if len(output) >= limit:
                break
        return output

    def show(self, limit: int = 20) -> None:
        """Print up to the given number of records from the dataset.

        Time complexity: O(limit specified)

        Args:
            limit: The max number of records to print.
        """
        for row in self.take(limit):
            print(row)

    def count(self) -> int:
        """Count the number of records in the dataset.

        Time complexity: O(dataset size / parallelism), O(1) for parquet

        Returns:
            The number of records in the dataset.
        """

        @ray.remote
        def count(block: Block[T]) -> int:
            return block.num_rows()

        return sum(ray.get([count.remote(block) for block in self._blocks]))

    def sum(self) -> int:
        """Sum up the elements of this dataset.

        Time complexity: O(dataset size / parallelism)

        Returns:
            The sum of the records in the dataset.
        """

        @ray.remote
        def agg(block: Block[T]) -> int:
            return sum(block.iter_rows())

        return sum(ray.get([agg.remote(block) for block in self._blocks]))

    def schema(self) -> Union[type, "pyarrow.lib.Schema"]:
        """Return the schema of the dataset.

        For datasets of Arrow records, this will return the Arrow schema.
        For dataset of Python objects, this returns their Python type.

        Time complexity: O(1)

        Returns:
            The Python type or Arrow schema of the records.
        """
        raise NotImplementedError  # P0

    def num_blocks(self) -> int:
        """Return the number of blocks of this dataset.

        Time complexity: O(1)

        Returns:
            The number of blocks of this dataset.
        """
        return len(self._blocks)

    def size_bytes(self) -> int:
        """Return the in-memory size of the dataset.

        Time complexity: O(dataset size / parallelism), O(1) for parquet

        Returns:
            The in-memory size of the dataset in bytes.
        """
        raise NotImplementedError  # P0

    def input_files(self) -> List[str]:
        """Return the list of input files for the dataset.

        Time complexity: O(num input files)

        Returns:
            The list of input files used to create the dataset.
        """
        raise NotImplementedError  # P0

    def write_parquet(path: str,
                      filesystem: Optional["pyarrow.fs.FileSystem"] = None
                      ) -> None:
        """Write the dataset to parquet.

        This is only supported for datasets convertible to Arrow records.

        Examples:
            >>> ds.repartition(num_files).write_parquet("s3://bucket/path")

        Time complexity: O(dataset size / parallelism)

        Args:
            path: The path in the filesystem to write to.
            filesystem: The filesystem implementation to write to.
        """
        raise NotImplementedError  # P0

    def write_json(path: str,
                   filesystem: Optional["pyarrow.fs.FileSystem"] = None
                   ) -> None:
        """Write the dataset to json.

        This is only supported for datasets convertible to Arrow records.

        Examples:
            >>> ds.repartition(num_files).write_json("s3://bucket/path")

        Time complexity: O(dataset size / parallelism)

        Args:
            path: The path in the filesystem to write to.
            filesystem: The filesystem implementation to write to.
        """
        raise NotImplementedError  # P0

    def write_csv(path: str,
                  filesystem: Optional["pyarrow.fs.FileSystem"] = None
                  ) -> None:
        """Write the dataset to csv.

        This is only supported for datasets convertible to Arrow records.

        Examples:
            >>> ds.repartition(num_files).write_csv("s3://bucket/path")

        Time complexity: O(dataset size / parallelism)

        Args:
            path: The path in the filesystem to write to.
            filesystem: The filesystem implementation to write to.
        """
        raise NotImplementedError  # P0

    def iter_rows(self, prefetch_blocks: int = 0) -> Iterator[T]:
        """Return a local row iterator over the dataset.

        Examples:
            >>> for i in ray.data.range(1000000).iter_rows():
            ...     print(i)

        Time complexity: O(1)

        Args:
            prefetch_blocks: The number of blocks to prefetch ahead of the
                current block during the scan.

        Returns:
            A local iterator over the entire dataset.
        """

        for block in self.iter_batches(prefetch_blocks=prefetch_blocks):
            for row in block.iter_rows():
                yield row

    def iter_batches(self,
                     prefetch_blocks: int = 0,
                     batch_size: int = None,
                     batch_format: str = "pandas") -> Iterator[BatchType]:
        """Return a local batched iterator over the dataset.

        Examples:
            >>> for pandas_df in ray.data.range(1000000).iter_batches():
            ...     print(pandas_df)

        Time complexity: O(1)

        Args:
            prefetch_blocks: The number of blocks to prefetch ahead of the
                current block during the scan.
            batch_size: Record batch size, or None to let the system pick.
            batch_format: Specify "pandas" to select ``pandas.DataFrame`` as
                the batch format, or "pyarrow" to select ``pyarrow.Table``.

        Returns:
            A list of iterators over record batches.
        """

        if prefetch_blocks > 0:
            raise NotImplementedError  # P1

        for b in self._blocks:
            block = ray.get(b)
            yield block

    def to_torch(self, **todo) -> "ray.util.sgd.torch.TorchMLDataset":
        """Return a dataset that can be used for Torch distributed training.

        Time complexity: O(1)

        Returns:
            A TorchMLDataset.
        """
        raise NotImplementedError  # P1

    def to_tf(self, **todo) -> "ray.util.sgd.tf.TFMLDataset":
        """Return a dataset that can be used for TF distributed training.

        Time complexity: O(1)

        Returns:
            A TFMLDataset.
        """
        raise NotImplementedError  # P1

    def to_dask(self) -> "dask.DataFrame":
        """Convert this dataset into a Dask dataframe.

        Time complexity: O(1)

        Returns:
            A Dask dataframe created from this dataset.
        """
        raise NotImplementedError  # P1

    def to_modin(self) -> "modin.DataFrame":
        """Convert this dataset into a Modin dataframe.

        Time complexity: O(1)

        Returns:
            A Modin dataframe created from this dataset.
        """
        raise NotImplementedError  # P1

    def to_pandas(self) -> List[ObjectRef["pandas.DataFrame"]]:
        """Convert this dataset into a set of Pandas dataframes.

        Time complexity: O(1)

        Returns:
            A list of remote Pandas dataframes created from this dataset.
        """
        raise NotImplementedError  # P1

    def to_spark(self) -> "pyspark.sql.DataFrame":
        """Convert this dataset into a Spark dataframe.

        Time complexity: O(1)

        Returns:
            A Spark dataframe created from this dataset.
        """
        raise NotImplementedError  # P2

    def __repr__(self) -> str:
        return "Dataset({} blocks)".format(len(self._blocks))

    def __str__(self) -> str:
        return repr(self)

    def _block_sizes(self) -> List[int]:
        @ray.remote
        def query(block: Block[T]) -> int:
            return block.num_rows()

        return ray.get([query.remote(b) for b in self._blocks])

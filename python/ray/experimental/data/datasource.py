import builtins
from typing import Generic, List, Callable, Union, TypeVar

import ray
from ray.experimental.data.impl.arrow_block import ArrowRow, \
    DelegatingArrowBlockBuilder
from ray.experimental.data.impl.block import Block, ListBlock
from ray.experimental.data.impl.block_list import BlockList, BlockMetadata

T = TypeVar("T")
R = TypeVar("WriteResult")


class Datasource(Generic[T]):
    """Interface for defining a custom ``ray.data.Dataset`` datasource.

    To read a datasource into a dataset, use ``ray.data.read_datasource()``.
    To write to a writable datasource, use ``Dataset.write_datasource()``.

    See ``RangeDatasource`` and ``TestOutput`` below for examples of how to
    implement readable and writable datasources.
    """

    def prepare_read(self, parallelism: int,
                     **read_args) -> List["ReadTask[T]"]:
        """Return the list of tasks needed to perform a read.

        Args:
            parallelism: The requested read parallelism. The number of read
                tasks should be as close to this value as possible.
            read_args: Additional kwargs to pass to the datasource impl.

        Returns:
            A list of read tasks that can be executed to read blocks from the
            datasource in parallel.
        """
        raise NotImplementedError

    def prepare_write(self, blocks: BlockList,
                      **write_args) -> List["WriteTask[T, R]"]:
        """Return the list of tasks needed to perform a write.

        Args:
            blocks: List of data block references and block metadata. It is
                recommended that one write task be generated per block.
            write_args: Additional kwargs to pass to the datasource impl.

        Returns:
            A list of write tasks that can be executed to write blocks to the
            datasource in parallel.
        """
        raise NotImplementedError

    def on_write_complete(self, write_tasks: List["WriteTask[T, R]"],
                          write_task_outputs: List[R]) -> None:
        """Callback for when a write job completes.

        This can be used to "commit" a write output. This method must
        succeed prior to ``write_datasource()`` returning to the user. If this
        method fails, then ``on_write_failed()`` will be called.

        Args:
            write_tasks: The list of the original write tasks.
            write_task_outputs: The list of write task outputs.
        """
        pass

    def on_write_failed(self, write_tasks: List["WriteTask[T, R]"],
                        error: Exception) -> None:
        """Callback for when a write job fails.

        This is called on a best-effort basis on write failures.

        Args:
            write_tasks: The list of the original write tasks.
            error: The first error encountered.
        """
        pass


class ReadTask(Callable[[], Block[T]]):
    """A function used to read a block of a dataset.

    Read tasks are generated by ``datasource.prepare_read()``, and return
    a ``ray.data.Block`` when called. Metadata about the read operation can
    be retrieved via ``get_metadata()`` prior to executing the read.

    Ray will execute read tasks in remote functions to parallelize execution.
    """

    def __init__(self, read_fn: Callable[[], Block[T]],
                 metadata: BlockMetadata):
        self._metadata = metadata
        self._read_fn = read_fn

    def get_metadata(self) -> BlockMetadata:
        return self._metadata

    def __call__(self) -> Block[T]:
        return self._read_fn()


class WriteTask(Callable[[Block[T]], R]):
    """A function used to write a chunk of a dataset.

    Write tasks are generated by ``datasource.prepare_write()``, and return
    a datasource-specific output that is passed to ``on_write_complete()``
    on write completion.

    Ray will execute write tasks in remote functions to parallelize execution.
    """

    def __init__(self, write_fn: Callable[[Block[T]], R]):
        self._write_fn = write_fn

    def __call__(self) -> R:
        self._write_fn()


class RangeDatasource(Datasource[Union[ArrowRow, int]]):
    """An example datasource that generates ranges of numbers from [0..n).

    Examples:
        >>> source = RangeDatasource()
        >>> ray.data.read_datasource(source, n=10).take()
        ... [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def prepare_read(self, parallelism: int, n: int,
                     use_arrow: bool) -> List[ReadTask]:
        read_tasks: List[ReadTask] = []
        block_size = max(1, n // parallelism)

        def make_block(start: int, count: int) -> ListBlock:
            builder = DelegatingArrowBlockBuilder()
            for value in builtins.range(start, start + count):
                if use_arrow:
                    builder.add({"value": value})
                else:
                    builder.add(value)
            return builder.build()

        i = 0
        while i < n:
            count = min(block_size, n - i)
            if use_arrow:
                import pyarrow
                schema = pyarrow.Table.from_pydict({"value": [0]}).schema
            else:
                schema = int
            read_tasks.append(
                ReadTask(
                    lambda i=i, count=count: make_block(i, count),
                    BlockMetadata(
                        num_rows=count,
                        size_bytes=8 * count,
                        schema=schema,
                        input_files=None)))
            i += block_size

        return read_tasks


class TestOutput(Datasource[Union[ArrowRow, int]]):
    """An example implementation of a writable datasource for testing.

    Examples:
        >>> output = TestOutput()
        >>> ray.data.range(10).write_datasource(output)
        >>> assert output.num_ok == 1
    """

    def __init__(self):
        @ray.remote
        class DataSink:
            def __init__(self):
                self.rows_written = 0
                self.enabled = True

            def write(self, block: Block[T]) -> str:
                if not self.enabled:
                    raise ValueError("disabled")
                self.rows_written += block.num_rows()
                return "ok"

            def get_rows_written(self):
                return self.rows_written

            def set_enabled(self, enabled):
                self.enabled = enabled

        self.data_sink = DataSink.remote()
        self.num_ok = 0
        self.num_failed = 0

    def prepare_write(self, blocks: BlockList,
                      **write_args) -> List["WriteTask[T, R]"]:
        tasks = []
        for b in blocks:
            tasks.append(
                WriteTask(lambda b=b: ray.get(self.data_sink.write.remote(b))))
        return tasks

    def on_write_complete(self, write_tasks: List["WriteTask[T, R]"],
                          write_task_outputs: List[R]) -> None:
        assert len(write_task_outputs) == len(write_tasks)
        assert all(w == "ok" for w in write_task_outputs)
        self.num_ok += 1

    def on_write_failed(self, write_tasks: List["WriteTask[T, R]"],
                        error: Exception) -> None:
        self.num_failed += 1

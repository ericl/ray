from typing import TypeVar, List, Any

import tqdm

import ray
from ray.experimental.data.impl.block import Block, ObjectRef

T = TypeVar("T")


def simple_shuffle(block_cls: type, input_blocks: List[ObjectRef[Block[T]]],
                   output_num_blocks: int) -> List[ObjectRef[Block[T]]]:
    input_num_blocks = len(input_blocks)

    @ray.remote(num_returns=output_num_blocks)
    def shuffle_map(serialized: Any) -> Any:
        block = block_cls.deserialize(serialized)
        slice_sz = max(1, block.num_rows() // output_num_blocks)
        slices = []
        for i in range(output_num_blocks):
            slices.append(block.slice(i * slice_sz, (i + 1) * slice_sz))
        return [s.serialize() for s in slices]

    @ray.remote
    def shuffle_reduce(*mapper_outputs: List[Any]) -> Any:
        builder = block_cls.builder()
        assert len(mapper_outputs) == input_num_blocks
        for serialized in mapper_outputs:
            block = block_cls.deserialize(serialized)
            builder.add_block(block)
        return builder.build().serialize()

    map_bar = tqdm.tqdm(total=input_num_blocks, position=0)
    map_bar.set_description("Shuffle Map")
    reduce_bar = tqdm.tqdm(total=output_num_blocks, position=1)
    reduce_bar.set_description("Shuffle Reduce")

    shuffle_map_out = [shuffle_map.remote(block) for block in input_blocks]

    remaining = [x[0] for x in shuffle_map_out]
    while remaining:
        done, remaining = ray.wait(remaining, fetch_local=False)
        map_bar.update(len(done))

    shuffle_reduce_out = [
        shuffle_reduce.remote(
            *[shuffle_map_out[i][j] for i in range(input_num_blocks)])
        for j in range(output_num_blocks)
    ]

    remaining = shuffle_reduce_out
    while remaining:
        done, remaining = ray.wait(remaining, fetch_local=False)
        reduce_bar.update(len(done))

    map_bar.close()
    reduce_bar.close()
    return shuffle_reduce_out

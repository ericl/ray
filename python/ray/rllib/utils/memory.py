from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def tf_aligned(size, dtype, align=64):
    """Returns an array of a given size that is 64-byte aligned.

    The returned array can be efficiently copied into GPU memory by TensorFlow.
    """

    n = size * dtype.itemsize
    empty = np.empty(n + (align - 1), dtype=np.uint8)
    data_align = empty.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    output = empty[offset:offset + n].view(dtype)

    assert len(output) == size, len(output)
    assert output.ctypes.data % align == 0, output.ctypes.data
    return output


def concat_aligned(items):
    """Concatenate arrays, ensuring the output is 64-byte aligned.

    This should be used instead of np.concatenate() to improve performance
    when the output array is likely to be fed into TensorFlow.
    """

    if len(items) == 0:
        return []
    elif len(items) == 1:
        return items[0]

    dtype = items[0].dtype
    if dtype in [np.float32, np.float64]:
        flat = tf_aligned(sum(s.size for s in items), dtype)
        batch_dim = sum(s.shape[0] for s in items)
        new_shape = (batch_dim, ) + items[0].shape[1:]
        output = flat.reshape(new_shape)
        assert output.ctypes.data % 64 == 0, output.ctypes.data
        np.concatenate(items, out=output)
        return output

    return np.concatenate(items)

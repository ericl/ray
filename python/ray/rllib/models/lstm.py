from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""LSTM support for RLlib.

The main trick here is that we add the time dimension at the last moment.
The non-LSTM layers of the model see their inputs as one flat batch. Before
the LSTM cell, we pad the input and add the expected time dimension. Then, we
remove the padding and reshape it back into a flat vector.

See the add_time_dimension() and remove_time_dimension() functions below for
more info.
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from ray.rllib.models.misc import linear, normc_initializer
from ray.rllib.models.model import Model


def add_time_dimension(inputs, seq_lens):
    """Adds a time dimension to padded inputs.

    Arguments:
        inputs (Tensor): a dense batch of un-padded sequences. That is,
            for seq_lens=[1, 2, 2], then inputs=[A, B, B, C, C], where
            A, B, C are sequence elements.
        seq_lens (Tensor): the sequence lengths within the input batch,
            suitable for passing to tf.nn.dynamic_rnn().

    Returns:
        Reshaped tensor of shape [len(seq_lens), max(seq_lens), ...].
    """

    num_seqs = tf.size(seq_lens)
    max_seq_len = tf.reduce_max(seq_lens)
    padded_size = num_seqs * max_seq_len
    indices = tf.boolean_mask(
        tf.range(padded_size),
        tf.reshape(tf.sequence_mask(seq_lens), [padded_size]))
    padded = tf.scatter_nd(
        tf.expand_dims(indices, 1), inputs,
        [padded_size] + inputs.get_shape().as_list()[1:])
    return tf.reshape(
        padded,
        [num_seqs, max_seq_len] + inputs.get_shape().as_list()[1:])


def remove_time_dimension(inputs, seq_lens):
    """Removes the padding added by add_time_dimension().

    Arguments:
        inputs (Tensor): a padded batch of shape [B, T, ...]. That is, for
            for seq_lens=[1, 2, 2], then inputs=[[A, 0], [B, 0], [C, C]],
            where A, B, C are sequence elements and 0 denotes padding.
        seq_lens (Tensor): the sequence lengths within the input batch.

    Returns:
        Reshaped tensor of shape [sum(seq_lens), ...].
    """

    return tf.boolean_mask(inputs, tf.sequence_mask(seq_lens))


def chop_into_sequences(episode_ids, state_columns, max_seq_len):
    """Truncate and pad experiences into fixed-length sequences.

    Arguments:
        episode_ids (list): List of episode ids for each step.
        state_columns (list): List of arrays containing LSTM state values.
        max_seq_len (int): Max length of sequences before truncation.

    Returns:
        s_init (list): Initial states for each sequence, of shape
            [NUM_SEQUENCES, ...].
        seq_lens (list): List of sequence lengths, of shape [NUM_SEQUENCES].

    Examples:
        >>> s_init, seq_lens = chop_into_sequences(
                episode_id=[1, 1, 5, 5, 5, 5],
                state_columns=[[4, 5, 4, 5, 5, 5]],
                max_seq_len=3)
        >>> print(s_init)
        [[4, 4, 5]]
        >>> print(seq_lens)
        [2, 3, 1]
    """

    prev_id = None
    seq_lens = []
    seq_len = 0
    for eps_id in episode_ids:
        if (prev_id is not None and eps_id != prev_id) or \
                seq_len >= max_seq_len:
            seq_lens.append(seq_len)
            seq_len = 0
        seq_len += 1
        prev_id = eps_id
    if seq_len:
        seq_lens.append(seq_len)
    assert sum(seq_lens) == len(episode_ids)

    # Dynamically shrink max len as needed to optimize memory usage
    max_seq_len = max(seq_lens)

    initial_states = []
    for s in state_columns:
        s = np.array(s)
        s_init = []
        i = 0
        for l in seq_lens:
            s_init.append(s[i])
            i += l
        initial_states.append(np.array(s_init))

    return initial_states, np.array(seq_lens)


class LSTM(Model):
    """Adds a LSTM cell on top of some other model output.

    Uses a linear layer at the end for output.
    """

    def _build_layers(self, inputs, num_outputs, options):
        padded_inputs = add_time_dimension(inputs, self.seq_lens)

        # Setup the LSTM cell
        cell_size = options.get("lstm_cell_size", 256)
        lstm = rnn.BasicLSTMCell(cell_size, state_is_tuple=True)
        self.state_init = [
            np.zeros(lstm.state_size.c, np.float32),
            np.zeros(lstm.state_size.h, np.float32)
        ]

        # Setup LSTM inputs
        if self.state_in:
            c_in, h_in = self.state_in
        else:
            c_in = tf.placeholder(
                tf.float32, [None, lstm.state_size.c], name="c")
            h_in = tf.placeholder(
                tf.float32, [None, lstm.state_size.h], name="h")
            self.state_in = [c_in, h_in]

        # Setup LSTM outputs
        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            lstm,
            padded_inputs,
            initial_state=state_in,
            sequence_length=self.seq_lens,
            time_major=False,
            dtype=tf.float32)

        self.state_out = list(lstm_state)

        # Compute outputs
        last_layer = tf.reshape(
            remove_time_dimension(lstm_out, self.seq_lens), [-1, cell_size])
        logits = linear(last_layer, num_outputs, "action",
                        normc_initializer(0.01))
        return logits, last_layer

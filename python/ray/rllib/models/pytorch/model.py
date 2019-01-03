from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from ray.rllib.models.model import _restore_original_dimensions


class TorchModel(nn.Module):
    """Defines an abstract network model for use with RLlib / PyTorch."""

    def __init__(self, obs_space, num_outputs, options):
        """All custom RLlib torch models must support this constructor.

        Arguments:
            obs_space (gym.Space): Input observation space.
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Dictionary of model options.
        """
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self.options = options

    def forward(self, input_dict, hidden_state):
        """Wraps _forward() to unpack flattened Dict and Tuple observations."""
        input_dict["obs"] = input_dict["obs"].float()  # TODO(ekl): avoid cast
        input_dict = _restore_original_dimensions(
            input_dict, self.obs_space, tensorlib=torch)
        print("I", hidden_state)
        outputs, features, vf, h = self._forward(input_dict, hidden_state)
        print("O", h)
        if type(hidden_state) is list or type(h) is list:
            raise DeprecationWarning(
                "List-type RNN state output is deprecated. Please use a "
                "single tensor value instead for hidden_state.")
        return outputs, features, vf, h

    def state_init(self):
        """Returns the initial hidden state tensor value, if any."""
        return torch.zeros(0)

    def _forward(self, input_dict, hidden_state=None):
        """Forward pass for the model.

        Prefer implementing this instead of forward() directly for proper
        handling of Dict and Tuple observations.

        Arguments:
            input_dict (dict): Dictionary of tensor inputs, commonly
                including "obs", "prev_action", "prev_reward", each of shape
                [BATCH_SIZE, ...].
            hidden_state (obj): Hidden state tensor, of shape
                [BATCH_SIZE, h_size], or None.

        Returns:
            (outputs, feature_layer, values, state): Tensors of size
                [BATCH_SIZE, num_outputs], [BATCH_SIZE, desired_feature_size],
                [BATCH_SIZE], and [BATCH_SIZE, h_size].
        """
        raise NotImplementedError

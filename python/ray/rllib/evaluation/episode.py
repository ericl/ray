from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import queue
import random

import numpy as np


class MultiAgentEpisode(object):
    """Tracks the current state of a (possibly multi-agent) episode.

    The APIs in this class should be considered experimental, but we should
    avoid changing things for the sake of changing them since users may
    depend on them for advanced algorithms.

    Use case 1: Model-based rollouts in multi-agent:
        A custom compute_actions() function in a policy graph can inspect the
        current episode state and perform a number of rollouts based on the
        policies and state of other agents in the environment.

    Use case 2: Returning extra rollouts data.
        The model rollouts can be returned back to the sampler by calling:
        >>> batch = episode.new_batch_builder()
        >>> for each transition:
               batch.add_values(...)  # see sampler for usage
        >>> episode.extra_batches.add(batch.build_and_reset())
    """

    def __init__(self, policies, policy_mapping_fn, batch_builder_factory):
        self.batch_builder = batch_builder_factory()
        self.extra_batches = queue.Queue()
        self.new_batch_builder = batch_builder_factory
        self.total_reward = 0.0
        self.length = 0
        self.episode_id = random.randrange(2e9)
        self.agent_rewards = defaultdict(float)
        self._policies = policies
        self._policy_mapping_fn = policy_mapping_fn
        self._agent_to_policy = {}
        self._agent_to_rnn_state = {}
        self._agent_to_last_obs = {}
        self._agent_to_last_action = {}
        self._agent_to_last_pi_info = {}

    def policy_for(self, agent_id):
        """Returns the policy graph for the specified agent.

        If the agent is new, the policy mapping fn will be called to bind the
        agent to a policy for the duration of the episode.
        """

        if agent_id not in self._agent_to_policy:
            self._agent_to_policy[agent_id] = self._policy_mapping_fn(agent_id)
        return self._agent_to_policy[agent_id]

    def last_observation_for(self, agent_id):
        """Returns the last observation for the specified agent."""

        return self._agent_to_last_obs.get(agent_id)

    def last_action_for(self, agent_id):
        """Returns the last action for the specified agent."""

        action = self._agent_to_last_action[agent_id]
        # Concatenate tuple actions
        if isinstance(action, list):
            action = np.concatenate(action, axis=0).flatten()
        return action

    def rnn_state_for(self, agent_id):
        """Returns the last RNN state for the specified agent."""

        if agent_id not in self._agent_to_rnn_state:
            policy = self._policies[self.policy_for(agent_id)]
            self._agent_to_rnn_state[agent_id] = policy.get_initial_state()
        return self._agent_to_rnn_state[agent_id]

    def last_pi_info_for(self, agent_id):
        """Returns the last info object for the specified agent."""

        return self._agent_to_last_pi_info[agent_id]

    def add_agent_rewards(self, reward_dict):
        for agent_id, reward in reward_dict.items():
            if reward is not None:
                self.agent_rewards[agent_id,
                                   self.policy_for(agent_id)] += reward
                self.total_reward += reward

    def set_rnn_state(self, agent_id, rnn_state):
        self._agent_to_rnn_state[agent_id] = rnn_state

    def set_last_observation(self, agent_id, obs):
        self._agent_to_last_obs[agent_id] = obs

    def set_last_action(self, agent_id, action):
        self._agent_to_last_action[agent_id] = action

    def set_last_pi_info(self, agent_id, pi_info):
        self._agent_to_last_pi_info[agent_id] = pi_info

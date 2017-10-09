from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import random
import sys


import numpy as np
import ray
import time
import yaml
import ray.rllib.ppo as ppo
import ray.rllib.es as es
import ray.rllib.dqn as dqn
import ray.rllib.a3c as a3c
import ray.rllib.external_agent as external


# TODO(rliaw): Catalog for Agents (AgentCatalog.)
PENDING = 'PENDING'
RUNNING = 'RUNNING'
TERMINATED = 'TERMINATED'
ERROR = 'ERROR'

AGENTS = {
    'PPO': (ppo.PPOAgent, ppo.DEFAULT_CONFIG),
    'ES': (es.ESAgent, es.DEFAULT_CONFIG),
    'DQN': (dqn.DQNAgent, dqn.DEFAULT_CONFIG),
    'A3C': (a3c.A3CAgent, a3c.DEFAULT_CONFIG),
    'External': (external.ExternalAgent, external.DEFAULT_CONFIG),
}

class Experiment(object):
    def __init__(
            self, env, alg, stopping_criterion, cp_freq,
            out_dir, i, config, was_resolved, resources):
        self.alg = alg
        self.env = env
        self.config = config
        self.cp_freq = cp_freq
        self.was_resolved = was_resolved
        self.resources = resources
        # TODO(rliaw): Stopping criterion needs direction (min or max)
        self.stopping_criterion = stopping_criterion
        self.last_result = None
        self.checkpoint_path = None
        self.agent = None
        self.status = PENDING
        self.out_dir = out_dir
        self.num_gpus = resources.get('gpu', 0)
        self.i = i

    def checkpoint(self):
        path = ray.get(self.agent.save.remote())
        print("checkpointed at " + path)
        self.checkpoint_path = path
        return path

    def resource_requirements(self):
        return self.resources

    def start(self):
        self.status = RUNNING
        (agent_class, agent_config) = AGENTS[self.alg]
        config = agent_config.copy()
        for k in self.config.keys():
            if k not in config and self.alg != "External":
                raise Exception(
                    'Unknown agent config `{}`, all agent configs: {}'.format(
                        k, config.keys()))
        config.update(self.config)
        cls = ray.remote(num_gpus=self.resources.get('gpu', 0))(agent_class)
        # TODO(rliaw): make sure agent takes in SEED parameter
        self.agent = cls.remote(
            self.env, config, self.out_dir, 'trial_{}_{}'.format(
                self.i, self.param_str()))

    def stop(self):
        self.status = TERMINATED
        try:
            self.agent.stop.remote()
            self.agent.__ray_terminate__.remote(self.agent._ray_actor_id.id())
        except:
            print("Error stopping agent:", sys.exc_info()[0])
            self.status = ERROR
        finally:
            self.agent = None

    def train_remote(self):
        return self.agent.train.remote()

    def should_stop(self, result):
        # should take an arbitrary (set) of key, value specified by config
        return any(getattr(result, criteria) >= stop_value
                    for criteria, stop_value in self.stopping_criterion.items())

    def should_checkpoint(self):
        if self.cp_freq is None:
            return False
        if self.checkpoint_path is None:
            return True
        return (self.last_result.training_iteration) % self.cp_freq == 0

    def param_str(self):
        return '_'.join(
            [k + '=' + str(v) for k, v in self.config.items()
                if self.was_resolved[k]])

    def progress_string(self):
        if self.last_result is None:
            return self.status
        return '{}, {} s, {} ts, {} itrs, {} rew'.format(
            self.status,
            int(self.last_result.time_total_s),
            int(self.last_result.timesteps_total),
            self.last_result.training_iteration + 1,
            round(self.last_result.episode_reward_mean, 1))

    def update_progress(self, new_result):
        self.last_result = new_result

    def __str__(self):
        identifier = '{}_{}_{}'.format(self.alg, self.env, self.i)
        params = self.param_str()
        if params:
            identifier += '_' + params
        return identifier

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

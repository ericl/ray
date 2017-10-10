#!/usr/bin/env python

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
from ray.rllib.hpsearch.experiment import Experiment
from ray.rllib.hpsearch.experiment_runner import ExperimentRunner
from ray.rllib.hpsearch.utils import gpu_count


class GridSearch(object):
    def __init__(self, agent_cfg):
        self.cfg = agent_cfg
        self.grid_values = []
        for p, val in agent_cfg.items():
            if type(val) == dict and 'grid_search' in val:
                assert type(val['grid_search'] == list)
                self.grid_values.append((p, val['grid_search']))
        self.value_indices = [0] * len(self.grid_values)

    def next(self):
        cfg = self.cfg.copy()
        was_resolved = {}
        for i, (k, values) in enumerate(self.grid_values):
            idx = self.value_indices[i]
            cfg[k] = values[idx]
            self._increment(i)
            was_resolved[k] = True
        return cfg, was_resolved

    def _increment(self, i):
        self.value_indices[i] += 1
        if self.value_indices[i] >= len(self.grid_values[i]):
            self.value_indices[i] = 0
            if i + 1 < len(self.value_indices):
                self._increment(i + 1)


def parse_configuration(yaml_file):
    ''' Parses yaml_file for specifying experiment setup
        and return Experiment objects, one for each trial '''
    with open(yaml_file) as f:
        configuration = yaml.load(f)

    experiments = []

    def resolve(agent_cfg, was_resolved, i):
        ''' Resolves issues such as distributions and such '''
        assert type(agent_cfg) == dict
        cfg = agent_cfg.copy()
        for p, val in cfg.items():
            if type(val) == dict and 'eval' in val:
                cfg[p] = eval(val['eval'], {
                    'random': random,
                    'np': np,
                }, {
                    '_i': i,
                })
                was_resolved[p] = True
        return cfg, was_resolved

    for exp_name, exp_cfg in configuration.items():
        if 'search' in configuration:
            np.random.seed(exp_cfg['search']['search_seed'])
        env_name = exp_cfg['env']
        alg_name = exp_cfg['alg']
        cp_freq = exp_cfg.get('checkpoint_freq')
        stopping_criterion = exp_cfg['stop']
        out_dir = '/tmp/rllib/' + exp_name
        os.makedirs(out_dir, exist_ok=True)
        grid_search = GridSearch(exp_cfg['parameters'])
        for i in range(exp_cfg['max_trials']):
            grid_out, was_resolved = grid_search.next()
            resolved, was_resolved = resolve(grid_out, was_resolved, i)
            experiments.append(Experiment(
                env_name, alg_name, stopping_criterion, cp_freq, out_dir, i,
                resolved, was_resolved, exp_cfg['resources']))

    return experiments

if __name__ == '__main__':
    experiments = parse_configuration(sys.argv[1])
    ray.init(num_gpus=gpu_count())
    runner = ExperimentRunner(experiments)

    # TODO(ekl) implement crash recovery from status files
    
    def debug_print(title='Status'):
        print('== {} ==\n{}'.format(title, runner.debug_string()))
        print('Tensorboard dir: {}'.format(experiments[0].out_dir))
        print()

    debug_print('Starting')
    assert runner.can_launch_more(), \
        "Not enough resources for even one experiment"

    while not runner.is_finished():
        while runner.can_launch_more():
            runner.launch_experiment()
            debug_print()
        runner.process_events()
        debug_print()
    debug_print('Completed')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import ray
from ray.tune import run_experiments, grid_search
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.ppo import PPOAgent

class MyPreprocessorClass(Preprocessor):
    def _init(self):
        self.shape = self._obs_space.shape
        np.random.seed(self._options["custom_options"]["seed"])
        self.out_size = self._options["custom_options"]["out_size"]
        self.noise_size = self._options["custom_options"]["noise_size"]
        self.A = np.random.rand(self.out_size, 4 + self.noise_size)
        self.shape = (self.out_size,)

    def transform(self, observation):
        noise = np.random.rand(self.noise_size)
        return np.matmul(self.A, np.concatenate([observation, noise]))

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

ray.init()
run_experiments({
    "cartpole": {
        "run": "PPO",
        "env": "CartPole-v0",
        "repeat": 1,
        "resources": {
            "cpu": 3,
        },
        "stop": {
            "episode_reward_mean": 200,
            "time_total_s": 300,
        },
        "config": {
            "num_sgd_iter": 10,
            "num_workers": 2,
            "model": {
                "custom_preprocessor": "my_prep",
                "custom_options": {
                    "seed": 0,
                    "noise_size": grid_search([10, 50]),
                    "out_size": grid_search([4, 10, 100, 1000]),
                },
            },
        },
    }
})

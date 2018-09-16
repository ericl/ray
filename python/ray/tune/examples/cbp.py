import numpy as np
import os

import ray
from ray.tune import Trainable, run_experiments
from ray.tune.suggest import HyperOptSearch
from ray.tune.schedulers import CheckpointBasedPruning, AsyncHyperBandScheduler
from hyperopt import hp


class EasyModel(Trainable):
    def _setup(self):
        self.cur_reward = 0.0

    def _train(self):
        r = np.random.uniform(0, 1)
        a = 1.0
        for k, v in self.config.items():
            if k.startswith("a_"):
                a *= v
        alpha = min(1, self.cur_reward / 25.0)
        self.cur_reward += a * alpha + r * (1 - alpha)
        return {"episode_reward_mean": self.cur_reward}

    def _save(self, checkpoint_dir):
        path = checkpoint_dir + "/state-{}".format(self._iteration)
        with open(path, "w") as f:
            f.write(str(self.cur_reward))
        return path

    def _restore(self, checkpoint_path):
        self.cur_reward = float(open(checkpoint_path).read().strip())


if __name__ == "__main__":
    ray.init(num_cpus=1)

    space = {
        "a_0": hp.uniform("a_0", 0, 1),
        "a_1": hp.uniform("a_1", 0, 1),
        "a_2": hp.uniform("a_2", 0, 1),
    }
    algo = HyperOptSearch(
        space,
        max_concurrent=4,
        reward_attr="episode_reward_mean")
    
    cbp = CheckpointBasedPruning(
        time_attr="training_iteration",
        reltime_attr="iterations_since_restore",
        reward_attr="episode_reward_mean",
        checkpoint_eval_t=2,
        checkpoint_min_reward=20,
#        bootstrap_checkpoint=os.path.abspath("state-70"),
        reduction_factor=25)
    
    hb = AsyncHyperBandScheduler(
       time_attr="training_iteration",
       reward_attr="episode_reward_mean",
       max_t=100,
       grace_period=10,
       reduction_factor=3,
       brackets=3)

    run_experiments({
        "easy2": {
            "run": EasyModel,
            "num_samples": 300,
            "stop": {
                "training_iteration": 100,
            },
            "config": {
                "a_0": lambda _: np.random.uniform(0.0, 1),
                "a_1": lambda _: np.random.uniform(0.0, 1),
                "a_2": lambda _: np.random.uniform(0.0, 1),
            },
        }
    }, scheduler=cbp) #search_alg=algo) #scheduler=cbp)

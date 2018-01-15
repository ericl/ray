from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.tune import register_env, run_experiments

from env import CarlaEnv, ENV_CONFIG
from models import register_carla_model
from scenarios import TOWN2_STRAIGHT

env_name = "carla_env"
env_config = ENV_CONFIG.copy()
env_config.update({
    "verbose": False,
    "x_res": 80,
    "y_res": 80,
    "framestack": 1,
    "discrete_actions": False,
    "squash_action_logits": False,
    "server_map": "/Game/Maps/Town02",
    "scenarios": TOWN2_STRAIGHT,
})

register_env(env_name, lambda env_config: CarlaEnv(env_config))
register_carla_model()

run_experiments({
    "carla-bc": {
        "run": "BC",
        "env": "carla_env",
        "resources": {"cpu": 1},
        "config": {
            "dataset_path": "~/Desktop/AgentHuman/SeqTrain",
            "dataset_type": "hdf5",
            "env_config": env_config,
            "model": {
                "custom_model": "carla",
                "custom_options": {
                    "command_mode": "concat",
                    "image_shape": [80, 80, 3],
                },
                "conv_filters": [
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [512, [10, 10], 1],
                ],
            },
            "num_workers": 1,
        },
    },
})

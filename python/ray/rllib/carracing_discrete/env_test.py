import gym
import ray
from ray.tune.registry import register_env
from ray.tune import run_experiments
from ray.rllib import a3c
from atari_wrappers import NoopResetEnv, WarpFrame, FrameStack
from wrapper import DiscreteCarRacing

def build_racing_env(_):
    env = gym.make('CarRacing-v0')
    env = DiscreteCarRacing(env)
    env = NoopResetEnv(env)
    env.override_num_noops = 50
    env = WarpFrame(env, 80)
    env = FrameStack(env, 4)
    # hack needed to fix rendering on CarRacing-v0
    env = gym.wrappers.Monitor(env, "/tmp/rollouts", force=True)
    return env

env_creator_name = "discrete-carracing-v0"
register_env(env_creator_name, build_racing_env)

if __name__ == '__main__':
    ray.init()
    run_experiments({
            "demo": {
                "run": "A3C",
                "env": "discrete-carracing-v0",
                "config": {
                    "num_workers": 1,
                    "optimizer": {
                        "grads_per_step": 1000,
                    },
                }
            },
        })


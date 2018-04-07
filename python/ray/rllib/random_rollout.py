#!/usr/bin/env python
# xvfb-run -s '+iglx -screen 0 2048x2048x24' python random_rollout.py --env CarRacing-v0 --steps 500 --out car.json --image-out ~/Desktop/car

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import json
import os
import pickle

from scipy.misc import imsave

import gym
import ray
from ray.rllib.agent import get_agent_class
from ray.rllib.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.atari_wrappers import wrap_deepmind
from ray.rllib.utils.compression import pack
from ray.tune.registry import get_registry

EXAMPLE_USAGE = """
example usage:
    ./rollout.py --env CartPole-v0 --steps 1000000 --out rollouts.json
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Roll out a reinforcement learning agent "
                "given a checkpoint.", epilog=EXAMPLE_USAGE)

required_named = parser.add_argument_group("required named arguments")
required_named.add_argument(
    "--env", type=str, help="The gym environment to use.")
parser.add_argument(
    "--steps", default=None, help="Number of steps to roll out.")
parser.add_argument(
    "--out", default=None, help="Output filename.")
parser.add_argument(
    "--image-out", default=None, help="Output images to dir.")


def encode(obj):
    if isinstance(obj, np.ndarray):
        if len(obj) > 10:
            return pack(obj.copy()).decode("utf-8")
        else:
            return obj.tolist()
    else:
        return obj


def save_image(data, dest, i):
    if not os.path.exists(dest):
        os.makedirs(dest)
    imsave(os.path.join(dest, str(i) + ".png"), data)


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.env:
        if not args.config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = args.config.get("env")

    num_steps = int(args.steps)

    env = gym.make(args.env)
    env = wrap_deepmind(env)
    env = gym.wrappers.Monitor(env, "/tmp/rollouts", force=True)
    out = open(args.out, "w")
    steps = 0
    while steps < (num_steps or steps + 1):
        rollout = []
        state = env.reset()
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            if args.image_out:
                save_image(state[..., -1], args.image_out, steps)
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            out.write(json.dumps({
                "obs": encode(state),
                "new_obs": encode(next_state),
                "action": encode(action),
                "done": done,
                "timestep": steps,
                "reward": reward,
            }))
            out.write("\n")
            steps += 1
            state = next_state
        print("Episode reward", reward_total)

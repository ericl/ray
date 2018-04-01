#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import json
import os
import pickle

import gym
import ray
from ray.rllib.agent import get_agent_class
from ray.rllib.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog
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

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.env:
        if not args.config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = args.config.get("env")

    num_steps = int(args.steps)

    env = gym.make(args.env)
    out = open(args.out, "w")
    steps = 0
    while steps < (num_steps or steps + 1):
        rollout = []
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            action = random.choice(range(env.action_space.n))
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            out.write(json.dumps({
                "obs": state.tolist(),
                "new_obs": next_state.tolist(),
                "action": action,
                "done": done,
                "timestep": steps,
                "reward": reward,
            }))
            out.write("\n")
            steps += 1
            state = next_state
        print("Episode reward", reward_total)

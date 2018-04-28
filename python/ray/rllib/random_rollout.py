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
from ray.rllib.carracing_discrete import env_test
import gym
import ray
from ray.rllib.agent import get_agent_class
from ray.rllib.dqn.common.wrappers import wrap_dqn
from ray.rllib.models import ModelCatalog
from ray.rllib.cartpole import ImageDecoder
from ray.tune.registry import register_env
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
required_named.add_argument(
    "--repeat-prob", default=0, type=float, help="Probability of repeating.")
parser.add_argument(
    "--steps", default=None, help="Number of steps to roll out.")
parser.add_argument(
    "--out", default=None, help="Output filename.")
parser.add_argument(
    "--image-out", default=None, help="Output images to dir.")
parser.add_argument(
    "--restore", default="", help="Restore from checkpoint")
parser.add_argument(
    "--decode-model", default="", help="Decode using this model")


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


def get_repeat(repeat_prob):
    if random.random() < repeat_prob:
        return 50
    else:
        return 0


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.env:
        if not args.config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = args.config.get("env")
    
    num_steps = int(args.steps)
    agent = None
    if args.env == "car":
        env = env_test.build_racing_env(0) 
        register_env("car", env_test.build_racing_env)
    else:
        env = gym.make(args.env)

    if args.restore:
        ray.init()
        assert args.decode_model
        cls = get_agent_class("A3C")
        ModelCatalog.register_custom_preprocessor("img_decoder", ImageDecoder)
        model_opts = {
            "custom_preprocessor": "img_decoder",
            "custom_options": {
                "decode_model": args.decode_model,
                "h_size": 32,
            },
        }
        config = {
                "num_workers": 1,
                "model": model_opts,
        }
        agent = cls(config=config, env=args.env)
        agent.restore(args.restore)
        decoder = ImageDecoder(env.observation_space, model_opts)

    out = open(args.out, "w")
    steps = 0
    while steps < (num_steps or steps + 1):
        rollout = []
        repeat = 0
        state = env.reset()
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            if args.image_out:
                save_image(state[..., -1], args.image_out, steps)
            if repeat > 0:
                repeat -= 1
            else:
                repeat = get_repeat(args.repeat_prob)
                if repeat > 0 or not agent:
                    if args.env == "car":
                        action = 2
                    else:
                        action = env.action_space.sample()
                else:
                    decoded = decoder.transform(state)
                    action = int(agent.compute_action(decoded))
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            out.write(json.dumps({
                "obs": encode(state),
                "new_obs": encode(next_state),
                "action": encode(action),
                "done": done,
                "timestep": steps,
                "repeat": repeat,
                "reward": reward,
            }))
            out.write("\n")
            steps += 1
            state = next_state
        print("Episode reward", reward_total)


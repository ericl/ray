#!/usr/bin/env python

import random
import sys

import gym
from gym.envs.registration import register

import ray
from ray.tune.config_parser import make_parser, parse_to_trials
from ray.tune.trial_runner import TrialRunner
from ray.tune.trial import Trial


parser = make_parser("Test out Frozenlake")

parser.add_argument("--num-gpus", default=None, type=int,
                    help="Number of GPUs to allocate to Ray.")
parser.add_argument("--grid-size", default=4, type=int,
                    help="Size N of the NxN frozen lake grid.")
parser.add_argument("--hole-fraction", default=0.15, type=float,
                    help="Fraction of squares which are holes.")
parser.add_argument("--deterministic", default=True, type=bool,
                    help="Whether the env is deterministic.")


def env_creator(args, name):
    def make_lake():
        register(
            id=name,
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '8x8'},
            max_episode_steps=200,
            reward_threshold=0.99, # optimum = 1
        )
        env = gym.make(env)
        print(env.spec)
        return env

    return make_lake


def main(args):
    runner = TrialRunner()

    name = 'FrozenLake{}-{}x{}-v0'.format(
        args.deterministic and "Deterministic" or "",
        args.grid_size, args.grid_size)

    for _ in range(args.num_trials):
        runner.add_trial(
            Trial(
                env_creator(args, name), args.alg, args.config, args.local_dir,
                name, args.resources, args.stop, args.checkpoint_freq))

    while not runner.is_finished():
        runner.step()
        print(runner.debug_string())


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    ray.init(num_gpus=args.num_gpus)
    main(args)

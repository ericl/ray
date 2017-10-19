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
parser.add_argument("--grid-size", default=8, type=int,
                    help="Size N of the NxN frozen lake grid.")
parser.add_argument("--hole-fraction", default=0.15, type=float,
                    help="Fraction of squares which are holes.")
parser.add_argument("--deterministic", default=True, type=bool,
                    help="Whether the env is deterministic.")


def make_desc(grid_size, hole_fraction):
    random.seed(0)
    rows = []
    for y in range(grid_size):
        cells = ""
        for x in range(grid_size):
            if x == 0 and y == 0:
                cells += "S"
            elif x == grid_size - 1 and y == grid_size - 1:
                cells += "G"
            else:
                if random.random() < hole_fraction:
                    cells += "H"
                else:
                    cells += "F"
        rows.append(cells)
    return rows


def env_creator(args, name):
    desc = make_desc(args.grid_size, args.hole_fraction)
    print("== Frozen lake grid ==")
    for row in desc:
        print(row.replace("H", "^").replace("F", "."))

    def make_lake():
        register(
            id=name,
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={
                'desc': desc,
                'is_slippery': not args.deterministic,
            },
            max_episode_steps=200,
            reward_threshold=0.99, # optimum = 1
        )
        env = gym.make(name)
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

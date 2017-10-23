#!/usr/bin/env python

import random
import sys
import time

import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np

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
parser.add_argument("--one-hot", action='store_true',
                    help="Whether to one-hot encode the coordinates.")
parser.add_argument("--render", action='store_true',
                    help="Whether to periodically render episodes")
parser.add_argument("--test-increasing-sizes", action='store_true',
                    help="Whether to test different sized grids")


# TODO(ekl) why can't Ray pickle the class directly without a wrapper fn?
def wrap_render(env):
    class RenderSamples(gym.Wrapper):
        def __init__(self, env):
            super(RenderSamples, self).__init__(env)
            self.last_render_time = 0
            self.render = False

        def _step(self, action):
            res = self.env.step(action)
            if self.render:
                self.env.render()
            return res

        def _reset(self):
            if time.time() - self.last_render_time > 2:
                self.last_render_time = time.time()
                self.render = True
            else:
                self.render = False
            return self.env.reset()

    return RenderSamples(env)


def wrap_convert_cartesian(env, grid_size, one_hot):
    class ConvertToCartesianCoords(gym.ObservationWrapper):
        def __init__(self, env, grid_size, one_hot):
            super(ConvertToCartesianCoords, self).__init__(env)
            self.grid_size = grid_size
            self.one_hot = one_hot
            if self.one_hot:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self.grid_size * 2,))
            else:
                self.observation_space = spaces.Box(
                    low=0, high=grid_size, shape=(2,))

        def _observation(self, obs):
            x = obs % self.grid_size
            y = obs // self.grid_size
            if one_hot:
                new_obs = np.zeros(self.grid_size * 2)
                new_obs[x] = 1
                new_obs[self.grid_size + y] = 1
            else:
                new_obs = np.array((x, y))
            return new_obs

    return ConvertToCartesianCoords(env, grid_size, one_hot)


def wrap_reward_bonus(env, grid_size):
    class RewardBonus(gym.Wrapper):
        def __init__(self, env, grid_size):
            super(RewardBonus, self).__init__(env)
            self.obs = None
            self.grid_size = grid_size

        def _step(self, action):
            new_obs, rew, done, info = self.env.step(action)
            if done:
                # give a penalty for falling into a hole
                edge = self.grid_size - 1
                if new_obs[0] < edge or new_obs[1] < edge:
                    rew -= 1
                else:
                    rew += 7  # successful runs should have score of 10 total
            else:
                # give bonuses for partial progress
                rew += float(new_obs[0] - self.obs[0]) / (self.grid_size - 1)
                rew += float(new_obs[1] - self.obs[1]) / (self.grid_size - 1)
            self.obs = new_obs
            return new_obs, rew, done, info

        def _reset(self):
            self.obs = self.env.reset()
            return self.obs

    return RewardBonus(env, grid_size)


def make_desc(grid_size, hole_fraction):
    random.seed(0)
    rows = []
    for y in range(grid_size):
        cells = ""
        if hole_fraction > 0:
            hole_indices = list(range(grid_size))
            random.shuffle(hole_indices)
            hole_indices = hole_indices[
                :max(1, int(hole_fraction * grid_size))]
        else:
            hole_indices = []
        for x in range(grid_size):
            if x == 0 and y == 0:
                cells += "S"
            elif x == grid_size - 1 and y == grid_size - 1:
                cells += "G"
            else:
                if x < 2 and y < 2:
                    near_corners = True
                elif x > grid_size - 3 and y > grid_size - 3:
                    near_corners = True
                else:
                    near_corners = False
                if x > grid_size - 6 and y > grid_size - 6:
                    near_goal = True
                else:
                    near_goal = False
                if x in hole_indices and not near_corners:
                    cells += "H"
                elif near_goal:
                    cells += "F"
                else:
                    cells += "S"
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
        env = wrap_reward_bonus(
            wrap_convert_cartesian(
                gym.make(name), args.grid_size, args.one_hot),
            args.grid_size)
        if args.render:
            env = wrap_render(env)
        return env

    return make_lake


def main(args):
    runner = TrialRunner()

    name = 'FrozenLake{}-{}x{}-v0'.format(
        args.deterministic and "Deterministic" or "",
        args.grid_size, args.grid_size)

    for i in range(args.num_trials):
        import argparse
        if args.test_increasing_sizes:
            v = vars(args)
            v['grid_size'] += i
            a = argparse.Namespace(**v)
        else:
            a = args
        name = 'FrozenLake{}-{}x{}-v0'.format(
            args.deterministic and "Deterministic" or "",
            a.grid_size, a.grid_size)
        creator = env_creator(a, name)
        creator.env_name = name
        runner.add_trial(
            Trial(
                creator, args.alg, args.config, args.local_dir,
                None, args.resources, args.stop, args.checkpoint_freq))

    while not runner.is_finished():
        runner.step()
        print(runner.debug_string())


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    ray.init(num_gpus=args.num_gpus)
    main(args)

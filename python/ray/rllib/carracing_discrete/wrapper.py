import gym
from gym import Wrapper
from gym import spaces

NUM_ACTIONS = 5
ALLOWED_ACTIONS = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0.8],
    [0, 0, 0],
]

SPECIAL_ACTIONS = {
}


class DiscreteCarRacing(Wrapper):
    def __init__(self, env):
        super(DiscreteCarRacing, self).__init__(env)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
    def step(self, action):
        return self.env.step(ALLOWED_ACTIONS[action])



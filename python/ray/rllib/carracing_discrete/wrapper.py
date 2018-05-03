import gym
from gym import Wrapper
from gym import spaces

NUM_ACTIONS = 4
ALLOWED_ACTIONS = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0.8]
]

SPECIAL_ACTIONS = {
    100: [0, 0, 0],  # neutral TODO(ekl) add to main set of actions
}


class DiscreteCarRacing(Wrapper):
    def __init__(self, env):
        super(DiscreteCarRacing, self).__init__(env)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
    def step(self, action):
        if action in SPECIAL_ACTIONS:
            return self.env.step(SPECIAL_ACTIONS[action])
        return self.env.step(ALLOWED_ACTIONS[action])



import random

from game import AbstractGame

import ray


class PGameState(object):
  def __init__(self, depth, level_index, branching_factor, seed):
    self.depth = depth
    self.level_index = level_index
    self.branching_factor = branching_factor
    self.seed = seed

  def getScore(self):
    score = 0
    d = self.depth
    idx = self.level_index
    rnd = random.random()
    while d > 0:
      # TODO(ekl) update score using a hash function instead of abusing random.seed
      random.seed(hash((self.seed, idx, d)))
      if d % 2 == 0:
        score -= int(random.uniform(0, 100))
      else:
        score += int(random.uniform(0, 100))
      d -= 1
      idx = int(idx / self.branching_factor)
    random.seed(rnd)  # restore randomness
    return score

  def __repr__(self):
    return "PGameState(depth=%d, index=%d, score=%d)" % (
      self.depth, self.level_index, self.getScore())

  def __eq__(self, other):
    return self.depth == other.depth and \
      self.level_index == other.level_index and \
      self.branching_factor == other.branching_factor and \
      self.seed == other.seed

  def __hash__(self):
    return self.depth + self.level_index + self.branching_factor + self.seed


class PGame(AbstractGame):
  def __init__(self, branching_factor, tree_depth, seed):
    self.branching_factor = branching_factor
    self.tree_depth = tree_depth
    self.seed = seed

  def getInitialState(self):
    return PGameState(
      depth = 0,
      level_index = 0,
      branching_factor = self.branching_factor,
      seed = self.seed)

  def getActions(self, state):
    if state.depth < self.tree_depth:
      return range(self.branching_factor)
    else:
      return []

  def nextState(self, state, action):
    assert action in self.getActions(state)
    return PGameState(
      depth = state.depth + 1,
      level_index = state.level_index * self.branching_factor + action,
      branching_factor = self.branching_factor,
      seed = self.seed)


ray.register_class(PGame)
ray.register_class(PGameState, pickle=True)
ray.register_class(range, pickle=True)

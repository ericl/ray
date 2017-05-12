import random

from mcts import TreeSearch


class RandomAgent(object):
  def __init__(self, game, isFirstPlayer):
    self.game = game

  def getAction(self, s):
    return random.choice(self.game.getActions(s))


class TreeSearchAgent(object):
  def __init__(self, game, isFirstPlayer, params):
    self.game = game
    self.search = TreeSearch(game, params)
    self.isFirstPlayer = isFirstPlayer

  def getAction(self, s):
    outcomes = self.search.scoreActions(s, self.isFirstPlayer)
    outcomes.sort(key = lambda av: -av[1])
    return outcomes[0][0]

#!/usr/bin/env python3

import random

import matplotlib
import ray
import seaborn  # visualization
import time

import mcts
from pgame import PGame, PGameState
from agent import TreeSearchAgent, RandomAgent


def treeAgentFactory(params):
  def construct(game, isFirstPlayer):
    return TreeSearchAgent(game, isFirstPlayer, params)
  return construct


BASELINE_AGENT_CLASS = treeAgentFactory(params = {
  "budget": 100,
})
VISUALIZE_HISTO_NUM_BINS = 10
NUM_RUNS = 20


def runIteration(agentClass, seed):
  game = PGame(8, 30, seed = seed)
  # Maximizing agent
  player1 = agentClass(game, isFirstPlayer = True)
  # Minimizing agent (baseline)
  player2 = BASELINE_AGENT_CLASS(game, isFirstPlayer = False)
  s = game.getInitialState()
  while not game.isTerminal(s):
    a = player1.getAction(s)
    s = game.nextState(s, a)
    a = player2.getAction(s)
    s = game.nextState(s, a)
  score = s.getScore()
  if score > 0:
    print("Test agent wins", s)
  else:
    print("Baseline agent wins", s)
  return score


def runIterations(agentClass, num):
  results = []
  for i in range(num):
    start = time.time()
    results.append(runIteration(agentClass, i))
    delta = time.time() - start
    if i == 0:
      print("Estimated time per experiment: " + str(delta * num))
  return results


if __name__ == '__main__':
  if mcts.USE_RAY:
    ray.init()
  test_agents = [
    ('random', RandomAgent),
    ('baseline', BASELINE_AGENT_CLASS),
    ('400_serial', treeAgentFactory({
      "budget": 400,
    })),
    ('4x_100_parallel', treeAgentFactory({
      "budget": 100,
      "batchSize": 100,
      "batchParallelism": 4,
    })),
  ]
  results = []
  for label, a in test_agents:
    runs = runIterations(a, NUM_RUNS)
    results.append((label, runs))
    print(label, sum([1 for x in runs if x > 0]) / float(NUM_RUNS))
  print('--')
  for label, runs in results:
    p = seaborn.distplot(runs, bins = VISUALIZE_HISTO_NUM_BINS, label = label)
    p.legend()
    print(label, sum([1 for x in runs if x > 0]) / float(NUM_RUNS))
  matplotlib.pyplot.show()

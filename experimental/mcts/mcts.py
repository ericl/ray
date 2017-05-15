import os
import pickle
import random
import time

import numpy as np
import ray


USE_RAY = "USE_RAY" in os.environ


class TreeNode(object):
  """
  Represents a node in a MCTS tree. These nodes are also stored in a hash table
  nodeTable, which allows for tree statistics to be updated via random access.
  """

  def __init__(self, nodeTable, c, randomness, game, state, parent, actionIndex):

    assert state not in nodeTable
    nodeTable[state] = self

    self.nodeTable = nodeTable
    self.c = c
    self.randomness = randomness
    self.game = game
    self.state = state
    self.parent = parent
    self.actionIndex = actionIndex
    if parent is None:
      self.action = None
    else:
      self.action = parent.actions[actionIndex]
    self.actions = game.getActions(state)
    self.rewards = 0.0
    self.numVisits = 0
    self.children = [None] * len(self.actions)

  def isFullyExpanded(self):
    return None not in self.children

  def meanReward(self):
    return self.rewards / self.numVisits

  def ucb1(self):
    exploration_factor = self.c * np.sqrt(2 * np.log(self.parent.numVisits) / self.numVisits)
    bound = self.meanReward() + exploration_factor
    assert(not np.isnan(bound))
    return bound

  def expand(self):
    remaining_actions = [
      (i, a) for (i, a) in enumerate(self.actions) if self.children[i] is None]
    i, a = random.choice(remaining_actions)
    nextState = self.game.nextState(self.state, a)
    node = TreeNode(self.nodeTable, self.c, self.randomness, self.game, nextState, self, i)
    self.children[i] = node
    return node

  def getBestChild(self):
    if self.randomness > 0 and random.random() < self.randomness:
      return random.choice(self.children)
    ranked = sorted(
      [(n.ucb1(), n) for n in self.children], key=lambda t: -t[0])
    return ranked[0][1]

  def selectOrExpand(self):
    cur = self
    while not self.game.isTerminal(cur.state):
      if not cur.isFullyExpanded():
        return cur.expand()
      else:
        cur = cur.getBestChild()
    return cur

  def applyAndCollectRewardsRecursively(self, reward, rewardTable):

    if self.state not in rewardTable:
      rewardTable[self.state] = (self.parent and self.parent.state, self.actionIndex, reward, 1)
    else:
      ps, ai, prevReward, prevVisits = rewardTable[self.state]
      rewardTable[self.state] = (ps, ai, prevReward + reward, prevVisits + 1)

    self.rewards += reward
    self.numVisits += 1

    if self.parent:
      self.parent.applyAndCollectRewardsRecursively(reward, rewardTable)

  def mergeRewards(self, reward, numVisits):
    self.rewards += reward
    self.numVisits += numVisits

  def __repr__(self):
    return 'TreeNode(state=%s, visits=%s, ucb=%s)' % (
      self.state,
      self.numVisits,
      self.ucb1() if self.numVisits > 0 else 'nan')

  def treeString(self, depth = 0, maxDepth = 9999):
    if depth > maxDepth:
      return
    print("--" * depth + str(self))
    for c in self.children:
      if c:
        c.treeString(depth + 1, maxDepth)

  def treeStats(self, depth = 0):
    stats = {
      "distinctStates_" + str(depth): 1,
      "distinctStates_sum": 1,
    }
    childStats = [c.treeStats(depth + 1) for c in self.children if c]
    # merge counters from children via sum
    for cs in childStats:
      for k, v in cs.items():
        if k in stats:
          stats[k] += v
        else:
          stats[k] = v
    return stats

  def treeStatsString(self):
    stats = self.treeStats()
    i = 0
    out = "tree stats: "
    while ("distinctStates_" + str(i)) in stats:
      out += str(stats["distinctStates_" + str(i)]) + ", "
      i += 1
    out += str(stats["distinctStates_sum"]) + " states, "
    out += str(self.numVisits) + " visits"
    return out


class TreeSearch(object):
  def __init__(self, game, params):
    self.game = game
    self.accumulatedBudget = 0
    self.accumulatdTime = 0
    self.setParams(
      {
        "c": .5,
        "budget": 100,
        "batchSize": 10,
        "batchParallelism": 4,
        "randomness": 0.0,
        "prewarmIters": 0
      },
      params)

  def setParams(self, defaults, params):
    overrides = params.copy()
    for k, v in defaults.items():
      if k in overrides:
        setattr(self, k, overrides[k])
        del overrides[k]
      else:
        setattr(self, k, v)
    assert len(overrides) == 0, overrides

  def rollout(self, leaf):
    s = leaf.state
    while not self.game.isTerminal(s):
      s = self.game.nextState(s, random.choice(leaf.actions))
    return s.getScore()

  def applyCollectedRewards(self, rewardTable, nodeTable):
    for (state, (parentState, actionIndex, reward, numVisits)) in rewardTable.items():
      if state not in nodeTable:
        self.createNodes(nodeTable, rewardTable, parentState, actionIndex, state)
        assert state in nodeTable
      nodeTable[state].mergeRewards(reward, numVisits)

  def createNodes(self, nodeTable, rewardTable, parentState, actionIndex, state):
    if parentState:
      if parentState not in nodeTable:
        (grandParentState, parentActionIndex, _, _) = rewardTable[parentState]
        self.createNodes(
          nodeTable, rewardTable, grandParentState, parentActionIndex, parentState)
      parentNode = nodeTable[parentState]
    else:
      parentNode = None

    newNode = TreeNode(
      nodeTable,
      self.c,
      self.randomness,
      self.game,
      state,
      parentNode,
      actionIndex)
    if parentNode:
      parentNode.children[actionIndex] = newNode

  @ray.remote
  def runBatch(self, rootState, broadcastTable, rewardMultiplier, batchSize):
    localTable = pickle.loads(broadcastTable)
    localRoot = localTable[rootState]
    return self.runBatchLocally(localRoot, rewardMultiplier, batchSize)

  def runBatchLocally(self, localRoot, rewardMultiplier, batchSize):
    batchReward = {}
    for _ in range(batchSize):
      leaf = localRoot.selectOrExpand()
      score = self.rollout(leaf)
      if score > 0:
        leaf.applyAndCollectRewardsRecursively(1 * rewardMultiplier, batchReward)
      elif score < 0:
        leaf.applyAndCollectRewardsRecursively(-1 * rewardMultiplier, batchReward)
      else:
        leaf.applyAndCollectRewardsRecursively(0, batchReward)
    return batchReward

  def scoreActions(self, rootState, isMaximizingAgent):
    if isMaximizingAgent:
      rewardMultiplier = 1
    else:
      rewardMultiplier = -1
    nodeTable = {}
    globalRoot = TreeNode(nodeTable, self.c, self.randomness, self.game, rootState, None, None)
    i = 0
    budget = self.budget
    start = time.time()

    for _ in range(self.prewarmIters):
      self.runBatch(globalRoot, rewardMultiplier, 1)

    while i < budget:
      if self.batchParallelism > 1:
        if USE_RAY:
          broadcastTable = ray.put(pickle.dumps(nodeTable))
        else:
          broadcastTable = pickle.dumps(nodeTable)

      batchFutures = []
      for _ in range(self.batchParallelism):
        if USE_RAY:
          batchReward = self.runBatch.remote(
            self, rootState, broadcastTable, rewardMultiplier, self.batchSize)
          batchFutures.append(batchReward)
        else:
          if self.batchParallelism > 1:
            localTable = pickle.loads(broadcastTable)
            localRoot = localTable[rootState]
          else:
            localRoot = globalRoot
          batchFutures.append(self.runBatchLocally(localRoot, rewardMultiplier, self.batchSize))
      if USE_RAY:
        batchRewards = [ray.get(f) for f in batchFutures]
      else:
        batchRewards = batchFutures

      if self.batchParallelism > 1:
        for batchReward in batchRewards:
          self.applyCollectedRewards(batchReward, nodeTable)

      i += self.batchSize

    self.accumulatdTime += time.time() - start
    self.accumulatedBudget += budget * self.batchParallelism
    print(
      self.accumulatedBudget / self.accumulatdTime,
      "rollouts per second @ batch size", self.batchSize,
      "cores", self.batchParallelism,
      "budget", budget)
    print(globalRoot.treeStatsString())
#    root.treeString(maxDepth = 10)
    return [(c.action, c.meanReward()) for c in globalRoot.children]

ray.register_class(TreeSearch, pickle=True)
ray.register_class(TreeNode, pickle=True)

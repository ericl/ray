class AbstractGame(object):
  def getInitialState():
    pass

  def getActions(self, state):
    pass

  def nextState(self, state, action):
    pass

  def isTerminal(self, state):
    return not self.getActions(state)

  def getScore(self, state):
    assert self.isTerminal(state)

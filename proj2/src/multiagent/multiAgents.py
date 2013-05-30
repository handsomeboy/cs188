# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        currPos = currentGameState.getPacmanPosition()
        currGhostStates = currentGameState.getGhostStates()
        currGhostPositions = currentGameState.getGhostPositions()
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        currFood = currentGameState.getFood()
        walls = currentGameState.getWalls().asList()
        
        newFood = newFood.asList()
        foodFeature = 0
        ghostFeature = 0
        scaredFeature = 0
        distances = [0]
        distances = [distance(newPos, foodSpot,"manhattan") for foodSpot in newFood]
        
        sortedNewFood = sorted(newFood, key = lambda x: distance(newPos, x, "manhattan"))
        """ calculate the food component of the evaluation function"""
        foodGotten = 0
        if newPos in currFood.asList():
            foodGotten = 3
        wallCount = 0
        """if (newPos != newFood):
            dx = cmp((newPos[0] - newFood[0][0]),0)
            dy = cmp((newPos[1] - newFood[0][1]),0) 
        
            for i in range(abs(newPos[0] - newFood[0][0])):
                if (i*dx + newPos[0], newPos[1]) in newFood:
                    wallCount +=1
            for j in range(abs(newPos[1] - newFood[0][1])):
                if (newPos[0], newPos[1] +  dy*j) in newFood:
                    wallCount +=1"""
        if not distances:
            closestFood = 0
        else:
            closestFood = min(distances)
            
        if closestFood == 0:
            foodFeature =   foodGotten + 2.5
        else:
            foodFeature =   foodGotten + ( 1.0/closestFood)
        
        """ calculate the ghost component of the evaluation function"""
        if min([distance(newPos, ghostspot, "manhattan") for ghostspot in newGhostPositions]) == 0:
            ghostFeature = -100.0
        else:
            ghostFeature = 3 * -1.0 / min([distance(newPos, ghostspot,"manhattan") for ghostspot in newGhostPositions]) 
            if min([distance(newPos, ghostspot,"manhattan") for ghostspot in newGhostPositions])  < 6:
                ghostFeature = 5 * ghostFeature
            
            
        """ calculate the scared time component of the evaluation function"""
        return   foodFeature + ghostFeature
def distance(p1, p2, distanceType):
    if distanceType == 'manhattan':
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    if distanceType == 'euclidian':
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1])** 2) **.5
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        agentIndex = 0
        plyLen = gameState.getNumAgents()
        
        ply = range(plyLen)
        
        depth = self.depth
        
        agentOrder = ply * depth
        bestAction = ""
        currAgent = 1
        
        values = []
        actions = gameState.getLegalActions(0)
        actionValue = {}
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            currVal = getValue(nextState, depth, agentOrder, currAgent )
            actionValue[action] = currVal
            values.append(currVal)
            
        for actionKey in actionValue:
            if actionValue[actionKey] == max(values):
                bestAction = actionKey
        return bestAction
def getValue(state, remainingDepth, agentOrder, currAgent):
    if state.isLose() or state.isWin() or currAgent == len(agentOrder):
        return scoreEvaluationFunction(state)
    nextAgent = agentOrder[currAgent]
    if nextAgent == 0:
        return maxValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent)
    if nextAgent > 0:
        return minValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent)
def maxValue(state, remainingDepth, agentOrder, agent, currAgent):
    value = float("-inf")
    actions = state.getLegalActions(agent)
    for action in actions:
        successor = state.generateSuccessor(agent,action)
        value = max(value, getValue(successor,remainingDepth, agentOrder, currAgent + 1))
    return value

def minValue(state, remainingDepth, agentOrder, agent, currAgent):
    value = float("inf")
    actions = state.getLegalActions(agent)
    for action in actions:
        successor = state.generateSuccessor(agent,action)
        value = min(value, getValue(successor, remainingDepth, agentOrder, currAgent + 1))
    return value
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        plyLen = gameState.getNumAgents()
        ply = range(plyLen)
        depth = self.depth
        agentOrder = ply * depth
        bestAction = ""
        currAgent = 1
        values = []
        actions = gameState.getLegalActions(0)
        actionValue = {}
        alpha = float("-inf")
        beta = float("inf")
        currVal = float("-inf")
        bestValue = 0
        bestAction = actions[0]
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            currVal = max(currVal, ABgetValue(nextState, depth, agentOrder, currAgent, alpha, beta ))
            if currVal > beta:
                return action
            if currVal > bestValue:
                bestValue = currVal
                bestAction = action
            alpha = max(alpha, currVal)
        return bestAction

def ABgetValue(state, remainingDepth, agentOrder, currAgent, alpha, beta):
    if state.isLose() or state.isWin() or currAgent == len(agentOrder):
        return scoreEvaluationFunction(state)
    nextAgent = agentOrder[currAgent]
    if nextAgent == 0:
        return ABmaxValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent, alpha, beta)
    if nextAgent > 0:
        return ABminValue(state, remainingDepth -1,  agentOrder, nextAgent, currAgent, alpha,beta)
def ABmaxValue(state, remainingDepth, agentOrder, agent, currAgent, alpha, beta):
    value = float("-inf")
    actions = state.getLegalActions(agent)
    for action in actions:
        successor = state.generateSuccessor(agent,action)
        value = max(value, ABgetValue(successor,remainingDepth, agentOrder, currAgent + 1, alpha, beta))
        if value > beta:
            return value
        alpha = max(alpha, value)
    return value

def ABminValue(state, remainingDepth, agentOrder, agent, currAgent, alpha, beta):
    value = float("inf")
    actions = state.getLegalActions(agent)
    for action in actions:
        successor = state.generateSuccessor(agent,action)
        value = min(value, ABgetValue(successor, remainingDepth, agentOrder, currAgent + 1, alpha, beta))
        if value < alpha:
            return value
        beta = min(beta, value)
    return value
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        plyLen = gameState.getNumAgents()
        
        ply = range(plyLen)
        depth = self.depth
        agentOrder = ply * depth
        bestAction = ""
        currAgent = 1
        values = []
        actions = gameState.getLegalActions(0)
        actionValue = {}
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            currVal = self.EXPgetValue(nextState, depth, agentOrder, currAgent )
            actionValue[action] = currVal
            values.append(currVal)
        for actionKey in actionValue:
            if actionValue[actionKey] == max(values):
                bestAction = actionKey
        return bestAction
        
    def EXPgetValue(self, state, remainingDepth, agentOrder, currAgent):
        if state.isLose() or state.isWin() or currAgent == len(agentOrder):
            return betterEvaluationFunction(state)
        nextAgent = agentOrder[currAgent]
        if nextAgent == 0:
            return self.EXPmaxValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent)
        if nextAgent > 0:
            return self.expValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent)
    def EXPmaxValue(self, state, remainingDepth, agentOrder, agent, currAgent):
        value = float("-inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            successor = state.generateSuccessor(agent,action)
            value = max(value, self.EXPgetValue(successor,remainingDepth, agentOrder, currAgent + 1))
        return value
    def expValue(self, state, remainingDepth, agentOrder, agent, currAgent):
        value = 0
        actions = state.getLegalActions(agent)
        for action in actions:
            successor = state.generateSuccessor(agent,action)
            value += (1.0/len(actions))* self.EXPgetValue(successor, remainingDepth, agentOrder, currAgent + 1)
        return value
def betterEvaluationFunction(currentGameState):
    
    """Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>"""
    
    "*** YOUR CODE HERE ***" 
    score = 0

    walls = currentGameState.getWalls().asList()   
    food = currentGameState.getFood().asList()
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    scaredGhosts = [ghost for ghost in ghostStates if ghost.scaredTimer > 0]
    
    
    
    wFood, wGhost, wScaredGhost, wScaredTime = [2.0, -6.0, 8.0, 0.75]
    
    """check win/lose condition"""
    if currentGameState.isLose():
        return float("-inf")
    if currentGameState.isWin():
        return float("inf")
    """extract gameState score feature"""
    
    currScoreFeature = currentGameState.getScore()
    """ extract food features"""
    
    minFoodDist = min([util.manhattanDistance(pacmanPosition, foodPosition) for foodPosition in food])
    
    if minFoodDist == 1:
        wFood = 4.0
    foodFeature = 0
    foodFeature = 1.0/ minFoodDist
    
    """extract ghost features"""
    minGhostDist = min([util.manhattanDistance(pacmanPosition,ghostPosition) for ghostPosition in ghostPositions])
    ghostFeature = 0
    if minGhostDist < 5:
        wGhost = -12.0
    ghostFeature = 1.0 / minGhostDist
    
    """ extract scared ghost features"""
    scaredFeature = 0
    if scaredTimes[0] > 0:
        minScaredGhostDist = min([util.manhattanDistance(pacmanPosition,ghostPosition) for ghostPosition in ghostPositions])
        scaredFeature = 1.0 / minScaredGhostDist
    
    scaredTimeFeature = (1.0/minGhostDist) * scaredTimes[0] 
        
    score = currScoreFeature + wFood*foodFeature + wGhost*ghostFeature +wScaredGhost * scaredFeature + wScaredTime *scaredTimeFeature
    return score
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentIndex = 0
        plyLen = gameState.getNumAgents()
        
        ply = range(plyLen)
        depth = self.depth
        agentOrder = ply * depth
        bestAction = ""
        currAgent = 1
        values = []
        actions = gameState.getLegalActions(0)
        actionValue = {}
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            currVal = self.EXPgetValue(nextState, depth, agentOrder, currAgent )
            actionValue[action] = currVal
            values.append(currVal)
        for actionKey in actionValue:
            if actionValue[actionKey] == max(values):
                bestAction = actionKey
        return bestAction
        
    def EXPgetValue(self, state, remainingDepth, agentOrder, currAgent):
        if state.isLose() or state.isWin() or currAgent == len(agentOrder):
            return betterEvaluationFunction(state)
        nextAgent = agentOrder[currAgent]
        if nextAgent == 0:
            return self.EXPmaxValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent)
        if nextAgent > 0:
            return self.expValue(state, remainingDepth -1, agentOrder, nextAgent, currAgent)
    def EXPmaxValue(self, state, remainingDepth, agentOrder, agent, currAgent):
        value = float("-inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            successor = state.generateSuccessor(agent,action)
            value = max(value, self.EXPgetValue(successor,remainingDepth, agentOrder, currAgent + 1))
        return value
    def expValue(self, state, remainingDepth, agentOrder, agent, currAgent):
        value = 0
        actions = state.getLegalActions(agent)
        for action in actions:
            successor = state.generateSuccessor(agent,action)
            value += (1.0/len(actions))* self.EXPgetValue(successor, remainingDepth, agentOrder, currAgent + 1)
        return value


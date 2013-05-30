# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """

        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    
    
    "*** YOUR CODE HERE ***"
    from game import Directions
    
    fringe = util.Stack()
    start = Node(problem.getStartState())
    current = start
    fringe.push(start)
    closed = set()
    actionsTaken = []
    while not fringe.isEmpty():
        current = fringe.pop()
        actionsTaken = current.actions
        if problem.isGoalState(current.state):
            return actionsTaken
        if current.state not in closed:
            closed.add(current.state)
            for neighbors in problem.getSuccessors(current.state):
                nextAction = list(actionsTaken)
                nextAction.append(neighbors[1])
                neighborNode = Node(neighbors[0], nextAction)
                fringe.push(neighborNode)
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    fringe = util.Queue()
    start = Node(problem.getStartState())
    current = start
    fringe.push(start)
    closed = set()
    actionsTaken = []
    while not fringe.isEmpty():
        current = fringe.pop()
        actionsTaken = current.actions
        if problem.isGoalState(current.state):
            return actionsTaken
        if current.state not in closed:
            closed.add(current.state)
            for neighbors in problem.getSuccessors(current.state):
                nextAction = list(actionsTaken)
                nextAction.append(neighbors[1])
                neighborNode = Node(neighbors[0], nextAction)
                fringe.push(neighborNode)
    
def uniformCostSearch(problem):
    from game import Directions
    cost = 0
    actionsTaken = []
    start = Node(problem.getStartState(),actionsTaken,cost)
    current = start
    fringe = util.PriorityQueue()
    fringe.push(start, cost)
    closed = set()
    while not fringe.isEmpty():
        current = fringe.pop()
        actionsTaken = current.actions
        if problem.isGoalState(current.state):
            return actionsTaken
        if current.state not in closed:
            closed.add(current.state)
            for neighbors in problem.getSuccessors(current.state):
                newActionList = list(actionsTaken)
                newActionList.append(neighbors[1])
                nextNeighbor = Node(neighbors[0], newActionList, current.gn + neighbors[2])
                fringe.push(nextNeighbor, nextNeighbor.gn)
    
    
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic = nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    from game import Directions
    cost = 0
    actionsTaken = []
    start = Node(problem.getStartState(),actionsTaken,cost, heuristic(problem.getStartState(), problem))
    current = start
    fringe = util.PriorityQueue()
    fringe.push(start, cost)
    closed = set()
        
    while not fringe.isEmpty():
        current = fringe.pop()
        actionsTaken = current.actions
        if problem.isGoalState(current.state):
            return actionsTaken
        if current.state not in closed:
            closed.add(current.state)
            for neighbors in problem.getSuccessors(current.state):
                newActionList = list(actionsTaken)
                newActionList.append(neighbors[1])
                nextNeighbor = Node(neighbors[0], newActionList, current.gn + neighbors[2], heuristic(neighbors[0], problem))
                fringe.push(nextNeighbor, nextNeighbor.gn + nextNeighbor.hn)
    

    
class Node(object):
    def __init__(self, state = None, actions = [], gn = 0, hn = 0):
            self.state = state
            self.actions = actions
            self.gn = gn
            self.hn = hn
    def __eq__(self,other):
        if other == None:
            return false
        return(self.state == other.state) and (self.actions == other.actions) and (self.gn == other.gn) and (self.hn == other.hn)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

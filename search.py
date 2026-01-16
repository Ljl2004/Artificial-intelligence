# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """
    Q1 BFS
    Search the shallowest nodes in the search tree first.
    
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    """Search the node of least total cost first."""
    """
        Q1 BFS
        Search the shallowest nodes in the search tree first.
        """
    # 初始化队列和访问记录
    queue = util.Queue()
    visited = set()

    # 将起始状态和空路径入队
    startState = problem.getStartState()
    queue.push((startState, []))
    visited.add(startState)

    while not queue.isEmpty():
        currentState, actions = queue.pop()

        # 检查是否到达目标状态
        if problem.isGoalState(currentState):
            return actions

        # 扩展当前节点的所有后继节点
        for successor, action, stepCost in problem.getSuccessors(currentState):
            if successor not in visited:
                visited.add(successor)
                newActions = actions + [action]
                queue.push((successor, newActions))

    return []  # 如果没有找到路径，返回空列表



def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """Search the node of least total cost first."""
    # 初始化优先队列和访问记录
    # 使用优先队列，按累计成本排序
    priorityQueue = util.PriorityQueue()
    visited = {}  # 记录到达每个状态的最低成本

    # 将起始状态入队，成本为0
    startState = problem.getStartState()
    priorityQueue.push((startState, [], 0), 0)
    visited[startState] = 0

    while not priorityQueue.isEmpty():
        currentState, actions, currentCost = priorityQueue.pop()

        # 如果当前路径的成本不是最低的，跳过
        if currentState in visited and visited[currentState] < currentCost:
            continue

        # 检查是否到达目标状态
        if problem.isGoalState(currentState):
            return actions

        # 扩展当前节点的所有后继节点
        for successor, action, stepCost in problem.getSuccessors(currentState):
            newCost = currentCost + stepCost
            newActions = actions + [action]

            # 如果这是一个新状态，或者找到到达该状态成本更低的路径
            if successor not in visited or newCost < visited[successor]:
                visited[successor] = newCost
                priorityQueue.push((successor, newActions, newCost), newCost)

    return []  # 如果没有找到路径，返回空列表
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """
    Q2 A*
    Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # 初始化优先队列和访问记录
    priorityQueue = util.PriorityQueue()
    visited = set()

    # 将起始状态入队，优先级为启发式值
    startState = problem.getStartState()
    priorityQueue.push((startState, [], 0), 0 + heuristic(startState, problem))

    while not priorityQueue.isEmpty():
        currentState, actions, currentCost = priorityQueue.pop()

        # 如果已经访问过该状态，跳过
        if currentState in visited:
            continue
        visited.add(currentState)

        # 检查是否到达目标状态
        if problem.isGoalState(currentState):
            return actions

        # 扩展当前节点的所有后继节点
        for successor, action, stepCost in problem.getSuccessors(currentState):
            if successor not in visited:
                newCost = currentCost + stepCost
                newActions = actions + [action]
                # 优先级 = 实际代价 + 启发式值
                priority = newCost + heuristic(successor, problem)
                priorityQueue.push((successor, newActions, newCost), priority)

    return []  # 如果没有找到路径，返回空列表


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

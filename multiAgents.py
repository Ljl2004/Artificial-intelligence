# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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
    Your minimax agent (Question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def terminalTest(gameState, depth):
            return depth == self.depth or gameState.isWin() or gameState.isLose()

        def maximizer(currentState, depth):
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)

            maxScore = float("-inf")
            legalActions = currentState.getLegalActions(0)

            for action in legalActions:
                successor = currentState.generateSuccessor(0, action)
                score = minimizer(successor, depth, 1)
                if score > maxScore:
                    maxScore = score

            return maxScore

        def minimizer(currentState, depth, ghostIndex):
            if terminalTest(currentState, depth):
                return self.evaluationFunction(currentState)

            minScore = float("inf")
            legalActions = currentState.getLegalActions(ghostIndex)

            for action in legalActions:
                successor = currentState.generateSuccessor(ghostIndex, action)

                # 如果是最后一个幽灵，进入下一层（吃豆人的回合）
                if ghostIndex == currentState.getNumAgents() - 1:
                    score = maximizer(successor, depth + 1)
                else:
                    # 继续处理下一个幽灵
                    score = minimizer(successor, depth, ghostIndex + 1)

                if score < minScore:
                    minScore = score

            return minScore

        # 主逻辑
        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestScore = float("-inf")

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = minimizer(successor, 0, 1)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState, depth, alpha, beta):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:  # Terminal Test
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                maxvalue = max(maxvalue, minLevel(successor, currDepth, 1, alpha1, beta))
                if maxvalue > beta:
                    return maxvalue
                alpha1 = max(alpha1, maxvalue)
            return maxvalue

        # For all ghosts.
        def minLevel(gameState, depth, agentIndex, alpha, beta):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose():  # Terminal Test
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min(minvalue, maxLevel(successor, depth, alpha, beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1, minvalue)
                else:
                    minvalue = min(minvalue, minLevel(successor, depth, agentIndex + 1, alpha, beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1, minvalue)
            return minvalue

        # Alpha-Beta Pruning
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            # Next level is a min level. Hence calling min for successors of the root.
            score = minLevel(nextState, 0, 1, alpha, beta)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha value at root.
            if score > beta:
                return returnAction
            alpha = max(alpha, score)
        return returnAction

        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Used only for pacman agent hence agentindex is always 0.
        def maxLevel(gameState, depth):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:  # Terminal Test
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            totalmaxvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                maxvalue = max(maxvalue, expectLevel(successor, currDepth, 1))
            return maxvalue

        # For all ghosts.
        def expectLevel(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():  # Terminal Test
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            totalexpectedvalue = 0
            numberofactions = len(actions)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = maxLevel(successor, depth)
                else:
                    expectedvalue = expectLevel(successor, depth, agentIndex + 1)
                totalexpectedvalue = totalexpectedvalue + expectedvalue
            if numberofactions == 0:
                return 0
            return float(totalexpectedvalue) / float(numberofactions)

        # Root level action.
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        for action in actions:
            nextState = gameState.generateSuccessor(0, action)
            # Next level is a expect level. Hence calling expectLevel for successors of the root.
            score = expectLevel(nextState, 0, 1)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction

        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
          In my evaluation function I have divided the final score of the state in two parts
           1. When the ghosts are scared identified scaredTimes>0.
           2. Normal ghosts.
        Common evaluation score between both parts is the sum of the score for current score the steps
          for which the ghosts are scared, the reciprocal of the sum of food distance and number of foods eaten

          In the first case, from the sum I subtract the distance of the ghosts from current state
          and the number of power pellets, as the ghosts are currently in scared state. So closer pacman is to ghost better score

          In the second case since the ghosts are not scared hence distance to ghosts and number of power pellets
          are added to the sum.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    """ Manhattan distance to the foods from the current state """
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    """ Manhattan distance to each ghost from the current state"""
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos, pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum(ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)

    score += currentGameState.getScore() + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:
        score += sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else:
        score += sumGhostDistance + numberofPowerPellets
    return score


# Abbreviation
better = betterEvaluationFunction

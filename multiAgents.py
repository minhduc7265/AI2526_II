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
PARAMS = {'food': 15.889136242364984, 'capsule': 12.868591992871977, 'ghost': 2.366298825966193, 'scared': 175, 'A': 16.93574730124145, 'B': 57.93619436594343}

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, f) for f in foodList)
            score += 10.0 / minFoodDist
            score += 4.0 / len(foodList) 

        if action == 'Stop':
            score -= 50

        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            scared = newScaredTimes[i]

            if scared > 0:
                if scared > dist:
                    score += 150.0 / (dist + 0.1)
            else:
                if dist == 0:
                    score -= 9999
                elif dist <= 1:
                    score -= 800
                elif dist <= 3:
                    score -= 150.0 / dist
                elif dist <= 6:
                    score -= 20.0 / dist

        return score


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

    def getAction(self, gameState: GameState):
        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            successors = [state.generateSuccessor(agentIndex, a) for a in legalActions]
            scores = [minimax(s, nextDepth, nextAgent) for s in successors]

            if agentIndex == 0:
                return max(scores)  
            else:
                return min(scores)   

        legalActions = gameState.getLegalActions(0)
        scores = [minimax(gameState.generateSuccessor(0, a), self.depth, 1) for a in legalActions]
        return legalActions[scores.index(max(scores))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):

        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)

                v = max(v, minValue(successor, depth, 1, alpha, beta))

                if v > beta:
                    return v

                alpha = max(alpha, v)

            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)

                if agentIndex == state.getNumAgents() - 1:

                    v = min(v, maxValue(successor, depth + 1, alpha, beta))
                else:
                    v = min(v, minValue(successor, depth, agentIndex + 1, alpha, beta))

                if v < alpha:
                    return v

                beta = min(beta, v)

            return v


        bestAction = None
        bestScore = float("-inf")

        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)


            currentScore = minValue(nextState, 0, 1, alpha, beta)

            if currentScore > bestScore:
                bestAction = action
                bestScore = currentScore

            alpha = max(alpha, bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):

        def maxValue(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                v = max(v, expMinValue(successor, depth, 1))
            return v

        def expMinValue(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(agentIndex)
            totalScore = 0
            score = 0
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == successor.getNumAgents() - 1:
                    score = maxValue(successor, depth + 1)
                else:
                    score = expMinValue(successor, depth, agentIndex + 1)
                totalScore += score
            return totalScore / len(legalActions)

        bestAction = None
        bestScore = float("-inf")

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            currentScore = expMinValue(nextState, 0, 1)
            if currentScore >= bestScore:
                bestAction = action
                bestScore = currentScore

        return bestAction

def betterEvaluationFunction(currentGameState):
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    currentDir = currentGameState.getPacmanState().getDirection()
    if currentDir == Directions.STOP:
        score -= 50

    score -= PARAMS["A"] * len(foodList)
    score -= PARAMS["B"] * len(capsules)
 

    if len(foodList) > 0:
        listFoodDistance = [manhattanDistance(pos, foodPos) for foodPos in foodList]
        minDistance = min(listFoodDistance)
        score += PARAMS["food"] / (minDistance + 0.1)
    
    if len(capsules) > 0:
        listCapDistance = [manhattanDistance(pos, capPos) for capPos in capsules]
        minCapDist = min(listCapDistance)
        score += PARAMS["capsule"] / (minCapDist + 0.1)

    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostScaredTime = ghost.scaredTimer
        ghostDistance = manhattanDistance(pos, ghostPos)

        if ghostScaredTime == 0:
            if ghostDistance <= 1:
                score -= 2000
            elif ghostDistance <= 2:
                score -= 1000
            elif ghostDistance <= 5:
                score -= PARAMS["ghost"] / (ghostDistance + 0.1)
        else:
            if ghostDistance <= ghostScaredTime:
                score += PARAMS["scared"] / (ghostDistance + 0.1)



    return score + random.uniform(0, 0.00000001)

# Abbreviation
better = betterEvaluationFunction

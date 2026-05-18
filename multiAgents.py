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


from util import manhattanDistance, Queue
from game import Directions
import random, util

from game import Agent
from pacman import GameState

from util import manhattanDistance, Queue
from game import Directions
import random, util

from game import Agent
from pacman import GameState



# DEFAULT LEAF EVALUATOR  (module scope – required for util.lookup)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    Fallback leaf evaluator: raw game score.
    Used only when no better evaluator is explicitly specified.
    """
    return currentGameState.getScore()


#  Q1 – REFLEX AGENT

class ReflexAgent(Agent):

    def getAction(self, gameState: GameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos         = successorGameState.getPacmanPosition()
        newFood        = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [gs.scaredTimer for gs in newGhostStates]
        newCapsules    = successorGameState.getCapsules()
        walls          = successorGameState.getWalls()

        score = successorGameState.getScore()

        # ── STOP penalty ─────────────────────────────────────────────────────
        # Prevents oscillation / freezing when other scores tie.
        if action == Directions.STOP:
            score -= 80

        # ── Food gradient ─────────────────────────────────────────────────────
        # ── Food gradient ─────────────────────────────────────────────────────
        foodList = newFood.asList()

        # IMPORTANT:
        # capsules are also objectives.
        # If no normal food remains, Pacman must actively finish capsules.

        targets = foodList + newCapsules

        if targets:

            closestTarget = min(
                manhattanDistance(newPos, t)
                for t in targets
            )

            # stronger finish pressure when only capsules remain
            if not foodList and newCapsules:
                score += 45.0 / max(closestTarget, 1)
            else:
                score += 15.0 / max(closestTarget, 1)

        else:
            score += 500

        # ── Ghost interaction ─────────────────────────────────────────────────
        exits = openNeighbours(newPos, walls)   # escape routes available

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(newPos, ghostState.getPosition())

            if scaredTime > 0:

                # aggressive mode after capsule
                if scaredTime > dist:
                    score += 300.0 / max(dist, 1)
                else:
                    score += 80.0 / max(dist, 1)
            else:
                # Active: graduated danger
                if dist <= 1:
                    score -= 1000
                elif dist <= 2:
                    score -= 300
                elif dist <= 3:
                    score -= 80
                elif dist <= 4:
                    score -= 30

                # ESCAPE ROUTE PENALTY
                # If Pacman is in a dead-end / corridor AND a ghost is close,
                # that position is far more dangerous than open space.
                # exits == 1 → dead end, exits == 2 → corridor
            if dist <= 4 and exits <= 2:

                # corridor / dead-end danger amplification
                if exits == 1:
                    score -= 300
                else:
                    score -= 190

        # ── Capsule bonus ─────────────────────────────────────────────────────
        if newCapsules:

            closestCap = min(
                manhattanDistance(newPos, c)
                for c in newCapsules
            )

            # ghost nearby => emergency capsule mode
            ghostDanger = False

            for ghostState in newGhostStates:

                if ghostState.scaredTimer == 0:

                    ghostDist = manhattanDistance(
                        newPos,
                        ghostState.getPosition()
                    )

                    if ghostDist <= 4:
                        ghostDanger = True
                        break

            if ghostDanger:
                score += 40.0 / max(closestCap, 1)
            else:
                score += 8.0 / max(closestCap, 1)
        return score


class MultiAgentSearchAgent(Agent):
    """
    self.depth              – full plies to search
    self.evaluationFunction – leaf-node scorer (defaults to betterEvaluationFunction
                              which is much stronger than raw score)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


# ============================================================================
# Q2 – MINIMAX AGENT
# ============================================================================

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

# ============================================================================
# Q3 – ALPHA-BETA PRUNING AGENT
# ============================================================================

class AlphaBetaAgent(MultiAgentSearchAgent):
    
    def getAction(self, gameState: GameState):
        numAgents = gameState.getNumAgents()

        def alphaBeta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0 and depth == 0:
                return self.evaluationFunction(state)
            # Manage turns and tree depth
            legalActions = state.getLegalActions(agentIndex)
            nextAgent    = (agentIndex + 1) % numAgents
            nextDepth    = depth - 1 if nextAgent == 0 else depth #The depth of the tree depends on agentIndex

            if agentIndex == 0:                        # MAX
                value = float('-inf')
                for a in legalActions:
                    child = state.generateSuccessor(agentIndex, a)
                    value = max(value, alphaBeta(child, nextAgent, nextDepth, alpha, beta))
                    if value > beta:                   
                        return value
                    alpha = max(alpha, value)
                return value
            else:                                      # MIN
                value = float('inf')
                for a in legalActions:
                    child = state.generateSuccessor(agentIndex, a)
                    value = min(value, alphaBeta(child, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:                  
                        return value
                    beta = min(beta, value)
                return value

        # Root: track best action manually
        alpha      = float('-inf')
        beta       = float('inf')
        bestAction = None
        bestValue  = float('-inf')

        for action in gameState.getLegalActions(0):
            child = gameState.generateSuccessor(0, action)
            v = alphaBeta(child, 1, self.depth, alpha, beta)
            if v > bestValue:
                bestValue  = v
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction


# ============================================================================
# Q4 – EXPECTIMAX AGENT
# ============================================================================

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState):

        numAgents = gameState.getNumAgents()

        def expectimax(state, agentIndex, depth):

            # terminal state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # PACMAN 
            if agentIndex == 0:

                value = float('-inf')

                for action in legalActions:

                    successor = state.generateSuccessor(
                        agentIndex,
                        action
                    )

                    value = max(
                        value,
                        expectimax(
                            successor,
                            nextAgent,
                            nextDepth
                        )
                    )

                return value

            # GHOST 
            else:

                total = 0

                for action in legalActions:

                    successor = state.generateSuccessor(
                        agentIndex,
                        action
                    )

                    total += expectimax(
                        successor,
                        nextAgent,
                        nextDepth
                    )

                return total / len(legalActions)

        # =================================
        # ROOT
        # =================================

        legalActions = gameState.getLegalActions(0)

        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:

            successor = gameState.generateSuccessor(0, action)

            value = expectimax(
                successor,
                1,
                self.depth
            )

            # OPTIONAL:
            # nhẹ nhàng discourage STOP
            if action == Directions.STOP:
                value -= 5

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        
    
#  BFS DISTANCE  (wall-aware) for Q5

def bfsDistance(pos, target, walls):
    
    if pos == target:
        return 0
    visited = set()
    queue   = Queue()
    queue.push((pos, 0))
    visited.add(pos)
    while not queue.isEmpty():
        (x, y), dist = queue.pop()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            npos   = (nx, ny)
            if npos == target:
                return dist + 1
            if not walls[nx][ny] and npos not in visited:
                visited.add(npos)
                queue.push((npos, dist + 1))
    return float('inf')


def bfsClosest(pos, targets, walls):
    if not targets:
        return float('inf'), None
    targetSet = set(targets)
    if pos in targetSet:
        return 0, pos
    visited = set()
    queue   = Queue()
    queue.push((pos, 0))
    visited.add(pos)
    while not queue.isEmpty():
        (x, y), dist = queue.pop()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            npos   = (nx, ny)
            if npos in targetSet:
                return dist + 1, npos
            if not walls[nx][ny] and npos not in visited:
                visited.add(npos)
                queue.push((npos, dist + 1))
    return float('inf'), None


def openNeighbours(pos, walls):
    x, y = pos
    count = 0
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        if not walls[x+dx][y+dy]:
            count += 1
    return count
#  Q5 – BETTER EVALUATION FUNCTION  

def betterEvaluationFunction(currentGameState: GameState):
    # ── Terminal states dominate everything ──────────────────────────────────
    if currentGameState.isWin():
        return 100000 + currentGameState.getScore()
    if currentGameState.isLose():
        return -100000 + currentGameState.getScore()

    pacmanPos   = currentGameState.getPacmanPosition()
    foodList    = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules    = currentGameState.getCapsules()
    walls       = currentGameState.getWalls()

    # Base score game
    evaluation = currentGameState.getScore()

    # ── 1. Food count urgency ───
    evaluation -= 100 * len(foodList)

    # Strong linear suction to the nearest food pellet to maintain smoothness
    if foodList:
        bfsFoodDist, _ = bfsClosest(pacmanPos, foodList, walls)
        if bfsFoodDist < float('inf'):
            evaluation -= 2.0 * bfsFoodDist

    exits = openNeighbours(pacmanPos, walls)
    minActiveGhostDist = float('inf')

    # ── 2. Ghost Interaction & Tracking ────
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
        scaredTime = ghostState.scaredTimer

        bfsDist = bfsDistance(pacmanPos, ghostPos, walls)

        if scaredTime > 0:
            # FIERCE GHOST HUNTING
            if scaredTime > bfsDist:
                evaluation += (120.0 - 8.0 * bfsDist)
        else:
            minActiveGhostDist = min(minActiveGhostDist, bfsDist)
            # REALISTIC SURVIVAL MODE
            if bfsDist <= 1:
                evaluation -= 3000  
            elif bfsDist == 2:
                evaluation -= 800
            elif bfsDist == 3:
                evaluation -= 200
                # if the ghost is nearby and you end up in a narrow alley
                if exits <= 2:
                    evaluation -= 150

    # ── 3. Capsule urgency ───
    evaluation -= 200 * len(capsules)
    if capsules:
        capDist, _ = bfsClosest(pacmanPos, capsules, walls)
        if capDist < float('inf'):
            evaluation -= 3.0 * capDist
            if minActiveGhostDist <= 5 and capDist < minActiveGhostDist:
                # Force Pacman to charge straight into the capsule.
                evaluation += (160.0 - 20.0 * capDist)

    return evaluation

better = betterEvaluationFunction
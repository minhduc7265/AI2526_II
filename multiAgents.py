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



def scoreEvaluationFunction(currentGameState: GameState):
    """
    Fallback leaf evaluator: raw game score.
    Used only when no better evaluator is explicitly specified.
    """
    return currentGameState.getScore()


# Q1 – REFLEX AGENT

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

        # ── STOP penalty ──
        if action == Directions.STOP:
            score -= 80

        # ── Food gradient ──
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
            if dist <= 4 and exits <= 2:

                if exits == 1:
                    score -= 300
                else:
                    score -= 190

        # ── Capsule bonus ──
        if newCapsules:

            closestCap = min(
                manhattanDistance(newPos, c)
                for c in newCapsules
            )

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


#  BASE CLASS

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


# Q2: MINIMAX AGENT

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

#  Q3 – ALPHA-BETA PRUNING AGENT

class AlphaBetaAgent(MultiAgentSearchAgent):
    

    def getAction(self, gameState: GameState):
        numAgents = gameState.getNumAgents()

        def alphaBeta(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0 and depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent    = (agentIndex + 1) % numAgents
            nextDepth    = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:                        # MAX
                value = float('-inf')
                for a in legalActions:
                    child = state.generateSuccessor(agentIndex, a)
                    value = max(value, alphaBeta(child, nextAgent, nextDepth, alpha, beta))
                    if value > beta:                   # strict, NOT >=
                        return value
                    alpha = max(alpha, value)
                return value
            else:                                      # MIN
                value = float('inf')
                for a in legalActions:
                    child = state.generateSuccessor(agentIndex, a)
                    value = min(value, alphaBeta(child, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:                  # strict, NOT <=
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


#  Q4 – EXPECTIMAX AGENT

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

            # =========================
            # PACMAN (MAX)
            # =========================
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

            # =========================
            # GHOST (EXPECTATION)
            # =========================
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


#  Q5 – BETTER EVALUATION FUNCTION  

def betterEvaluationFunction(currentGameState: GameState):
    

    # ── Terminal states dominate everything ──
    if currentGameState.isWin():
        return 10000 + currentGameState.getScore()
    if currentGameState.isLose():
        return -10000 + currentGameState.getScore()

    pacmanPos   = currentGameState.getPacmanPosition()
    foodList    = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules    = currentGameState.getCapsules()
    walls       = currentGameState.getWalls()

    evaluation = currentGameState.getScore()

    # ── 1. Food count urgency ─────────────────────────────────────────────────
    evaluation -= 25 * len(foodList)

    # ── 2. Nearest food (BFS – wall-aware) ───────────────────────────────────
    bfsFoodDist, _ = bfsClosest(pacmanPos, foodList, walls)
    if bfsFoodDist < float('inf'):
        evaluation += 35.0 / max(bfsFoodDist, 1)

    # ── 3. Escape routes at current position ─────────────────────────────────
    exits = openNeighbours(pacmanPos, walls)
    # Mild bonus for being in open space (more tactical options)
    if exits >= 3:
        evaluation += 5

    # ── 4 & 5. Ghost interaction (BFS-accurate) ───────────────────────────────
    for ghostState in ghostStates:
        ghostPos   = ghostState.getPosition()
        scaredTime = ghostState.scaredTimer
        ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
        bfsDist = bfsDistance(pacmanPos, ghostPos, walls)
        if scaredTime > 0:
            # Scared: aggressive chasing
            # scaredTime > bfsDist means we can actually reach it before it recovers
            if scaredTime > bfsDist:
                evaluation += 150.0 / max(bfsDist, 1)
            else:
                evaluation += 50.0 / max(bfsDist, 1)   # too far, mild bonus
        else:
            # Active ghost: BFS-accurate graduated penalty
            if bfsDist <= 1:
                evaluation -= 800
            elif bfsDist <= 2:
                evaluation -= 220
            elif bfsDist <= 3:
                evaluation -= 100
            elif bfsDist <= 4:
                evaluation -= 25
            elif bfsDist >= 5:
                evaluation -= 10

            elif bfsDist >= 6:
                evaluation -= 15

            elif bfsDist >= 7:
                evaluation -= 20

            elif bfsDist >= 8:
                evaluation -= 30




            # ESCAPE ROUTE SCALING
            # Being in a dead-end or corridor with a nearby ghost is much more
            # dangerous than being in open space.
            if bfsDist <= 4 and exits <= 2:
                # exits == 1 (dead end): extra −200; exits == 2 (corridor): −100
                evaluation -= (3 - exits) * 150

    # ── 6. Capsule urgency ────────────────────────────────────────────────────

    if capsules:

        capDist, _ = bfsClosest(pacmanPos, capsules, walls)

        activeGhostNear = False

        for ghostState in ghostStates:

            if ghostState.scaredTimer == 0:

                ghostPos = ghostState.getPosition()

                # ghost positions may be float
                ghostPos = (int(ghostPos[0]), int(ghostPos[1]))

                gdist = bfsDistance(
                    pacmanPos,
                    ghostPos,
                    walls
                )

                if gdist <= 5:
                    activeGhostNear = True
                    break

        if activeGhostNear:
            evaluation += 120.0 / max(capDist, 1)
        else:
            evaluation += 20.0 / max(capDist, 1)

    evaluation -= 30 * len(capsules)
    return evaluation


# Abbreviation required by autograder
better = betterEvaluationFunction
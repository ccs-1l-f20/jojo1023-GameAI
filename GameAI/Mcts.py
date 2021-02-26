from Game import *
from GameAiConfig import *
import random
import math
import numpy
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Dict

class MctsNode:
    def __init__(self, priorProbability):
        self.currentPlayer = True
        self.visitCount = 0
        self.totalValue = 0
        self.priorProbability = priorProbability
        self.children = {}

    def MeanValue(self):
        if self.visitCount == 0:
            return 0
        else:
            return float(self.totalValue) / self.visitCount

class MctsStateEvaluator(ABC):
    @abstractmethod
    def StateEvaluation(self, game: Game) -> Tuple[Any, Dict[Any, float]]:
        pass

class RolloutMctsStateEvaluator(MctsStateEvaluator):
    def __init__(self):
        super().__init__()

    def StateEvaluation(self, game: Game) -> Tuple[Any, Dict[Any, float]]:
        moves = game.GetValidMoves()
        probDist = {}
        total = 0
        for a in moves:
            r = 1.0 / len(moves)  # random.uniform(0.01, 1)
            total += r
            probDist[a] = r
        for a in moves:
            probDist[a] /= total

        while not game.IsTerminal():
            moves = game.GetValidMoves()
            move = moves[random.randint(0, len(moves) - 1)]
            game.PlayMove(move)

        v = game.TerminalValue()
        # v = v if startPlayer else (-v)
        return v, probDist

class Mcts:
    def __init__(self, config: GameAiConfig, stateEvaluator: MctsStateEvaluator, verbose):
        self.config = config
        self.stateEvaluator = stateEvaluator
        self.verbose = verbose

    def runSims(self, game, addNoise=False):
        root = MctsNode(0)
        root.currentPlayer = game.GetCurrentPlayer()
        rootValue = 0
        for _ in range(self.config.numSimulations):
            cNode = root
            gameCpy = game.Clone()
            gamePath = [cNode]
            while len(cNode.children) > 0:
                move, cNode = self.pickChild(cNode)
                gameCpy.PlayMove(move)
                gamePath.append(cNode)

            if gameCpy.IsTerminal():
                value = gameCpy.TerminalValue()
                value = value # if cNode.currentPlayer else (-value)
            else:
                value = self.expandNode(cNode, gameCpy)
                if cNode == root:
                    if self.verbose:
                        print("NN value: " + str(value))
                    if addNoise:
                        self.AddExplorationNoise(root)

            for n in gamePath:
                v = value # if n.currentPlayer == cNode.currentPlayer else (-value)
                n.visitCount += 1
                n.totalValue += v

        maxVisit = -1
        maxAction = None
        newProbs = {}

        totalVisit = 0
        if self.verbose:
            print("NN MCTS value: " + str(root.MeanValue()))
        for a, n in root.children.items():
            if self.verbose:
                print("a: " + str(a) + " V: " + str(n.totalValue) + " VC: " + str(n.visitCount) +
                      " MV:" + str(n.MeanValue()))
            newProbs[a] = float(n.visitCount)
            totalVisit += n.visitCount
            if n.visitCount > maxVisit:
                maxVisit = n.visitCount
                maxAction = a
        for k in newProbs.keys():
            newProbs[k] /= float(totalVisit)
        return maxAction, newProbs

    def pickChild(self, node: MctsNode):
        mVal = None
        mAction = None
        mChild = None
        for a, child in node.children.items():
            usb = self.UsbScore(node, child)
            if mVal is None or mVal < usb:
                mVal = usb
                mAction = a
                mChild = child
        return mAction, mChild

    def UsbScore(self, parent: MctsNode, child: MctsNode):
        explore = math.log((parent.visitCount + self.config.usbExploreBase + 1) /
                           self.config.usbExploreBase) + self.config.usbExploreInit
        return (explore * child.priorProbability * (math.sqrt(parent.visitCount) / (1 + child.visitCount))) \
               + (child.MeanValue() if parent.currentPlayer else -child.MeanValue())
               # + (child.MeanValue() if parent.currentPlayer == child.currentPlayer else -child.MeanValue())

    def expandNode(self, node, game):
        node.currentPlayer = game.GetCurrentPlayer()
        availableMoves = game.GetValidMoves()
        value, probDist = self.stateEvaluation(game)
        total = 0.0
        for a in availableMoves:
            p = probDist[a]
            total += p
            node.children[a] = MctsNode(p)
        for a in availableMoves:
            node.children[a].priorProbability /= total
        return value

    def stateEvaluation(self, game: Game):
        return self.stateEvaluator.StateEvaluation(game)

    def AddExplorationNoise(self, node):
        actions = node.children.keys()
        noise = numpy.random.gamma(self.config.root_dirichlet_alpha, 1, len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].priorProbability = node.children[a].priorProbability * (1 - frac) + n * frac

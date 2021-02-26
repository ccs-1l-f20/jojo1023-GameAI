from typing import List, Any, Tuple
import numpy as np
import random
import keras
from Mcts import *
from keras.models import Model
from os import path
import threading
import multiprocessing
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Input
from keras.layers import concatenate
from keras.utils import plot_model
import pydot
from keras.optimizers import Adam
import pickle


class GameStateStorage:
    def __init__(self, currentPlayer, gameState, nextMoveProbs):
        self.currentPlayer = currentPlayer
        self.gameState = np.array(gameState)
        self.nextMoveProbs = np.array(nextMoveProbs)
        self.value = 0  # relative to current player


class GameStateBatch:
    def __init__(self, gameQueueSize, batchSize):
        self.gameQueueSize = gameQueueSize
        self.batchSize = batchSize
        self.gameGameStates = []
        self.amountOfGameStates = 0
        self.amountLock = multiprocessing.Lock()
        self.gamesLock = multiprocessing.Lock()

    def LoadGames(self):
        self.gameGameStates = pickle.load(open("savedGameStates.pkl", "rb"))
        for g in self.gameGameStates:
            self.amountOfGameStates += len(g)

    def SaveGames(self):
        with self.gamesLock:
            f = open("savedGameStates.pkl", "wb")
            print("Start Save")
            pickle.dump(self.gameGameStates, f)
            print("Done Save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def GetAmountOfGames(self):
        with self.gamesLock:
            return len(self.gameGameStates)

    def AddGameStates(self, gameStates: List[GameStateStorage]):

        with self.gamesLock:
            self.gameGameStates.append(gameStates)
            gamesAmount = len(self.gameGameStates)
        with self.amountLock:
            self.amountOfGameStates += len(gameStates)
        if gamesAmount > self.gameQueueSize:
            with self.gamesLock:
                lessGmStates = len(self.gameGameStates.pop(0))
            with self.amountLock:
                self.amountOfGameStates -= lessGmStates

    def GetAllBatch(self):
        inputs = []
        pOuts = []
        vOuts = []
        for g in self.gameGameStates:
            for gs in g:
                inputs.append(gs.gameState)
                pOuts.append(gs.nextMoveProbs)
                vOuts.append(gs.value)

        return np.array(inputs), np.array(pOuts), np.array(vOuts)

    def GetBatch(self):
        with self.amountLock:
            amtGameStates = self.amountOfGameStates
        with self.gamesLock:
            gameProbabilities = [len(g)/amtGameStates for g in self.gameGameStates]
            chosenGames = np.random.choice(np.arange(0, len(gameProbabilities)), self.batchSize, p=gameProbabilities)
        inputs = []
        pOuts = []
        vOuts = []
        for g in chosenGames:
            with self.gamesLock:
                cGame = self.gameGameStates[g]
                state = cGame[random.randint(0, len(cGame)-1)]
            inputs.append(state.gameState)
            pOuts.append(state.nextMoveProbs)
            vOuts.append(state.value)

        return np.array(inputs), np.array(pOuts), np.array(vOuts)


def GetNewNN(inputCount, outputCount):
    inL = Input(shape=(inputCount,))

    hidden = Dense(50, activation='tanh')(inL)
    hidden = Dense(50, activation='relu')(hidden)
    hidden = Dense(30, activation='tanh')(hidden)
    hidden = Dense(30, activation='relu')(hidden)

    hiddenV = Dense(15, activation='tanh')(hidden)
    hiddenV = Dense(5, activation='tanh')(hiddenV)
    outV = Dense(1, activation='tanh', name='valueOutput')(hiddenV)

    hiddenP = Dense(30, activation='relu')(hidden)
    hiddenP = Dense(30, activation='tanh')(hiddenP)
    outP = Dense(outputCount, activation='softmax', name='policyOutput')(hiddenP)

    model = Model(inputs=inL, outputs=[outV, outP])
    print(model.summary())
    # plot_model(model, to_file='multiple_outputs.png')
    return model


class NNStorage:
    def __init__(self):
        self.currentNNPath = "currentNNPath.txt"
        self.lockObject = threading.Lock()
        self._bestModel = None
        self.currentEpoch = 0

    def LoadBestModel(self, inputCount, outputCount):
        model = GetNewNN(inputCount, outputCount)
        if path.exists(self.currentNNPath):
            currentNNWeightsFilePath = open(self.currentNNPath, "r")
            filePathData = currentNNWeightsFilePath.read()
            currentNNWeightsFilePath.close()
            splitData = filePathData.split(":")
            self.currentEpoch = int(splitData[0])
            model.load_weights(splitData[1])
            cloneModel = keras.models.clone_model(model)
            cloneModel.set_weights(model.get_weights())
            cloneModel.make_predict_function()
            with self.lockObject:
                self._bestModel = cloneModel
        return model

    def CreateNewModel(self, inputCount, outputCount):
        model = GetNewNN(inputCount, outputCount)
        model.make_predict_function()
        with self.lockObject:
            self._bestModel = model

    def BestModelPredict(self, x):
        with self.lockObject:
            bm = self._bestModel
        return bm.predict(x=x)

    def HasBestModel(self):
        with self.lockObject:
            retVal = self._bestModel is not None
        return retVal

    def GetBestModelCopy(self) -> Model:
        with self.lockObject:
            cloneModel = keras.models.clone_model(self._bestModel)
            cloneModel.set_weights(self._bestModel.get_weights())
            return cloneModel

    def NewModel(self, model: Model, epoch, data: str):
        if path.exists(self.currentNNPath):
            currentNNWeightsFilePath = open(self.currentNNPath, "w+")
        else:
            currentNNWeightsFilePath = open(self.currentNNPath, "a+")
        nnPath = "Models/model-" + data + "-E" + str(epoch) + ".h5"
        model.save_weights(nnPath)
        currentNNWeightsFilePath.write(str(epoch) + ":" + nnPath)
        currentNNWeightsFilePath.close()

        cloneModel = keras.models.clone_model(model)
        cloneModel.set_weights(model.get_weights())
        cloneModel.make_predict_function()
        with self.lockObject:
            self._bestModel = cloneModel

class NNStateEvaluator(MctsStateEvaluator):
    def __init__(self, nnStorage: NNStorage, allPossibleMoves: List[Any]):
        super().__init__()
        self.nnStorage = nnStorage
        self.allPossibleMoves = allPossibleMoves

    def StateEvaluation(self, game: Game) -> Tuple[Any, Dict[Any, float]]:
        # nn = self.nnStorage.GetBestModel()
        (v, pd) = self.nnStorage.BestModelPredict(np.array([game.GetBoardState(game.GetCurrentPlayer())]))
        # nn.predict(x=np.array([game.GetBoardState(game.GetCurrentPlayer())]), use_multiprocessing=True)
        probDistDict = {}
        # total = 0
        for i in range(0, len(pd[0])):
            p = pd[0][i]
            # total += p
            probDistDict[self.allPossibleMoves[i]] = p
        # for i in range(0, len(pd[0])):
        #     probDistDict[self.allPossibleMoves[i]] /= total
        value = v[0][0] if game.GetCurrentPlayer() else -v[0][0]
        return value, probDistDict


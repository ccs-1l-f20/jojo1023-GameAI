from Mcts import *
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Input
from keras.layers import concatenate
from keras.utils import plot_model
import pydot
import os.path
from os import path
from keras.optimizers import Adam
from keras.optimizers import SGD


class GameStateStorage:
    def __init__(self, currentPlayer, gameState, nextMoveProbs):
        self.currentPlayer = currentPlayer
        self.gameState = gameState
        self.nextMoveProbs = nextMoveProbs
        self.value = 0  # relative to current player


class AlphaZero:
    def __init__(self, config: GameAiConfig):
        self.config = config
        self.currentNNPath = "currentNNPath.txt"

    def Start(self, game):
        gameGameStates = []
        allMoves = game.GetAllPossibleMoves()
        for i in range(0, 500):
            gameGameStates.append(self.PlayGame(game.Clone(), allMoves))

        self.TrainNN(gameGameStates)

    def PlayGame(self, game: Game, allMoves):
        mcts = Mcts(self.config, False)
        gameStates = []
        while not game.IsTerminal():  # add max length
            cp = game.GetCurrentPlayer()
            (a, probsDict) = mcts.runSims(game, addNoise=True)
            gs = game.GetBoardState(cp)
            probs = []
            for a in allMoves:
                if a in probsDict:
                    probs.append(probsDict[a])
                else:
                    probs.append(0)
            gameStates.append(GameStateStorage(cp, gs, probs))
            game.PlayMove(a)

        gameValue = game.TerminalValue()
        for gs in gameStates:
            gs.value = (gameValue if gs.currentPlayer else -gameValue)
        return gameStates

    def TrainNN(self, gameGameStates: list[list[GameStateStorage]]):
        if len(gameGameStates) <= 0 or len(gameGameStates[0]) <= 0:
            return
        model = self.GetNewNN(len(gameGameStates[0][0].gameState), len(gameGameStates[0][0].nextMoveProbs))
        if path.exists(self.currentNNPath):
            currentNNWeightsFilePath = open(self.currentNNPath, "w+")
        else:
            currentNNWeightsFilePath = open(self.currentNNPath, "a+")

        opt = SGD(learning_rate=self.config.learning_rate_schedule[0], momentum=self.config.momentum)
        lrScheduleCallback = keras.callbacks.LearningRateScheduler(self.config.learning_rate_schedule)

        model.compile(optimizer=opt, loss={'valueOutput': 'mse', 'policyOutput': 'categorical_crossentropy'},
                      metrics={'valueOutput': 'mae', 'policyOutput': 'accuracy'})

        model.train_on_batch(x=sdf, y=dsf)
        currentNNWeightsFilePath.close()
    def GetNNBatch(self, gameGameStates: list[list[GameStateStorage]]):
        pass


    def GetNewNN(self, inputCount, outputCount):
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


az = AlphaZero(GameAiConfig())
db = DotsAndBoxes(3)
az.TrainNN(None)
az.GetNewNN(len(db.GetBoardState(True)), len(db.GetAllPossibleMoves()))

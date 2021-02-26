import queue
from typing import List, Any, Tuple
from AlphaZeroUtils import *
from Mcts import *
import keras
from keras.optimizers import SGD
import numpy as np
import threading
import time
import multiprocessing


# def initialize_shared(q):
#     global gameStateQueue
#     gameStateQueue=q
doneGames = 0
def PlayAlphaZeroGame(config: GameAiConfig, game: Game, stateEvaluator,
                      index=-1, stopThreadPtr=None):
    checkForStopThread = stopThreadPtr is not None
    allMoves = game.GetAllPossibleMoves()
    print("Pre Model: " + str(index))
    # if self.nnStorage.HasBestModel():
    #     stateEvaluator = NNStateEvaluator(self.nnStorage, allMoves)
    # else:
    #     stateEvaluator = RolloutMctsStateEvaluator()
    print("Pre MCTS: " + str(index))
    mcts = Mcts(config, stateEvaluator, False)
    print("Post MCTS: " + str(index))
    gameStates = []
    turnCount = 0
    while not game.IsTerminal():  # add max length
        if checkForStopThread and stopThreadPtr[0]:
            return None
        cp = game.GetCurrentPlayer()
        print("Move: " + str(turnCount) + " I: " + str(index))
        (a, probsDict) = mcts.runSims(game, addNoise=True)
        gs = game.GetBoardState(cp)
        probs = []
        for m in allMoves:
            if m in probsDict:
                probs.append(probsDict[m])
            else:
                probs.append(0)
        gameStates.append(GameStateStorage(cp, gs, probs))
        game.PlayMove(a)
        turnCount += 1

    gameValue = game.TerminalValue()
    for gs in gameStates:
        gs.value = (gameValue if gs.currentPlayer else -gameValue)
    doneGames+=1
    print("Done Games:" + str(doneGames))
    return gameStates


class AlphaZero:
    def __init__(self, config: GameAiConfig):
        self.config = config
        self.nnStorage = NNStorage()
        # self.stateEvaluator = RolloutMctsStateEvaluator()
        self.gameStateBatch = None
        self.stopThreads = [False]

    def Start(self, game):
        self.gameStateBatch = GameStateBatch(self.config.gameBatchQueueSize, self.config.batchSize)
        allMoves = game.GetAllPossibleMoves()
        inputCount = len(game.GetBoardState(game.GetCurrentPlayer()))
        outputCount = len(allMoves)
        self.nnStorage.LoadBestModel(inputCount, outputCount)
        self.gameStateBatch.LoadGames()
        # self.NNThread(inputCount, outputCount)
        # return
        processArgs = []
        pool = multiprocessing.Pool(10)  # , initializer=initialize_shared, initargs=(gameStatesQueue,)
        for i in range(0, 1000):
            allMovesCpy = game.GetAllPossibleMoves()
            thisProcessArgs = (self.config, game.Clone(), RolloutMctsStateEvaluator(), i, self.stopThreads,)
            processArgs.append(thisProcessArgs)
            # t = threading.Thread(target=self.PlayGameThread, args=(game.Clone(), i), daemon=True)
            # threads.append(t)
            # t.start()
        data = pool.starmap(PlayAlphaZeroGame, processArgs)
        pool.close()
        for d in data:
            self.gameStateBatch.AddGameStates(d)
        self.gameStateBatch.SaveGames()
        # for a in processArgs:
        #     a.join()

        # nnThread = threading.Thread(target=self.NNThread, args=(gameStatesQueue, inputCount, outputCount), daemon=True)
        # nnThread.start()

        # nnThread.join()
        # inputThread = threading.Thread(target=self.InputThread, daemon=True)
        # inputThread.start()
        # print("Done Starting Game Threads")

    def NNThread(self, inputCount, outputCount):
        amountOfGames = self.gameStateBatch.GetAmountOfGames()
        while amountOfGames < 1000:
            time.sleep(5)
            # while not gameStatesQueue.empty():
            #     gs = gameStatesQueue.get()
            #     self.gameStateBatch.AddGameStates(gs)

            amountOfGames = self.gameStateBatch.GetAmountOfGames()
            print("Amount of Games: " + str(amountOfGames) + "||||||||||||||||||||||||")

        self.TrainNN(self.gameStateBatch, self.nnStorage, inputCount, outputCount)

    def InputThread(self):
        while True:
            s = input()
            if s == 'save':
                self.stopThreads = [True]
                print("Threads Flagged to Stop !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                time.sleep(5)
                self.gameStateBatch.SaveGames()

    def PlayGameThread(self, game, index):
        for i in range(0, 100):
            allMoves = game.GetAllPossibleMoves()
            print("Start: " + str(index) + " Game: " + str(i))
            gameClone = game.Clone()
            returnMpValue = []
            p = multiprocessing.Process(target=PlayAlphaZeroGame,
                                        args=(self.config, gameClone, allMoves, RolloutMctsStateEvaluator(),
                                              returnMpValue, index, self.stopThreads), daemon=True)
            p.start()
            p.join()
            p.close()
            if self.stopThreads[0]:
                return
            # returnGmeStates = returnMpValue.value
            if returnMpValue is None:
                continue
            self.gameStateBatch.AddGameStates(returnMpValue)
            print("End: " + str(index) + " Game: " + str(i))

    def GetMove(self, game, verbose):
        allMoves = game.GetAllPossibleMoves()
        evaluator = NNStateEvaluator(self.nnStorage, allMoves) if self.nnStorage.HasBestModel() \
            else RolloutMctsStateEvaluator()
        mcts = Mcts(self.config, evaluator, verbose)

        (a, probsDict) = mcts.runSims(game, addNoise=True)
        return a

    def TrainNN(self, gameStateBatch: GameStateBatch, nnStorage: NNStorage, inputCount, outputCount):
        if not nnStorage.HasBestModel():
            nnStorage.CreateNewModel(inputCount, outputCount)

        model = nnStorage.GetBestModelCopy()

        lrSchedule = self.config.learning_rate_schedule
        maxLrIndex = 0
        for i in lrSchedule.keys():
            if nnStorage.currentEpoch >= i > maxLrIndex:
                maxLrIndex = i

        opt = SGD(learning_rate=lrSchedule[maxLrIndex], momentum=self.config.momentum)
        model.compile(optimizer=opt, loss={'valueOutput': 'mse', 'policyOutput': 'categorical_crossentropy'},
                      loss_weights={'valueOutput': 1, 'policyOutput': 1},
                      metrics={'valueOutput': 'mae', 'policyOutput': 'accuracy'})

        (inputs, pOut, vOut) = gameStateBatch.GetAllBatch()
        model.evaluate(x=inputs, y={'valueOutput': vOut, 'policyOutput': pOut})

        for i in range(nnStorage.currentEpoch, self.config.amountOfTrainingSteps):
            if i in lrSchedule:
                keras.backend.set_value(model.optimizer.learning_rate, lrSchedule[i])
            print("Pre Get Batch:" + str(i) + "----------------------------------------------------")
            (inputs, pOut, vOut) = gameStateBatch.GetBatch()
            print("Post Get Batch:" + str(i) + "----------------------------------------------------")
            model.fit(x=inputs, y={'valueOutput': vOut, 'policyOutput': pOut}, epochs=i + 1, initial_epoch=i,
                      batch_size=len(inputs), verbose=2)
            print("Post Fit Batch:" + str(i) + "----------------------------------------------------")
            print(keras.backend.eval(model.optimizer.lr))
            # print(model.evaluate(x=inputs, y={'valueOutput': vOut, 'policyOutput': pOut}))
            if i % self.config.checkpointInterval == 0:
                nnStorage.NewModel(model, i, "")
                # (inputs, pOut, vOut) = gameStateBatch.GetAllBatch()
                # model.evaluate(x=inputs, y={'valueOutput': vOut, 'policyOutput': pOut})
                print("Stored Model " + str(i))

# az = AlphaZero(GameAiConfig())
# db = DotsAndBoxes(3)
# az.TrainNN(None)
# az.GetNewNN(len(db.GetBoardState(True)), len(db.GetAllPossibleMoves()))

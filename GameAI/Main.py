from Game import *
from Mcts import *
from TkinterGui import *
from AlphaZero import *
import keras
import tensorflow as tf

if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    boardSize = 3
    aiIsTrue = False
    useNN = True
    training = True

    if training:
        az = AlphaZero(GameAiConfig())
        az.Start(DotsAndBoxes(boardSize))
    else:
        gui = DotsAndBoxesGui(boardSize, aiIsTrue, useNN)
        gui.run()

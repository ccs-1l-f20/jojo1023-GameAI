from Game import *
from Mcts import *
from TkinterGui import *


boardSize = 3
aiIsTrue = True
gui = DotsAndBoxesGui(boardSize, aiIsTrue)

gui.run()

from Game import *

db = DotsAndBoxes(4)

while True:
    inpt = str(input())
    db.PlayerMove(inpt)
    if db.IsTerminal():
        break

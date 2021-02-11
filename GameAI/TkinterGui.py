import tkinter as tk
import threading
from Mcts import *

class TkinterGui:
    def __init__(self):
        self.window = tk.Tk()
        self.InitGui()

    def run(self):
        self.window.mainloop()

    def InitGui(self):
        pass


class DotsAndBoxesGui(TkinterGui):
    def __init__(self, boardSize, aiIsTrue):
        self.boardSize = boardSize
        self.aiIsTrue = aiIsTrue
        self.boxes = []
        self.buttonsH = {}
        self.buttonsV = {}
        self.buttonsByMove = {}

        self.db = DotsAndBoxes(boardSize)
        self.mcts = Mcts(GameAiConfig(), True)
        self.defaultButtonColor = 'gray'
        self.disabledButtonColor = 'dim gray'
        super().__init__()

    def InitGui(self):
        gridSize = 2*self.boardSize + 1
        for i in range(1, gridSize, 2):
            self.window.columnconfigure(i, weight=100, minsize=100)
            self.window.rowconfigure(i, weight=100, minsize=100)
            rowBoxes = []
            for j in range(1, gridSize, 2):
                box = tk.Label(bg="white")
                box.grid(row=i, column=j, sticky="nsew", padx=5, pady=5)
                rowBoxes.append(box)
            self.boxes.append(rowBoxes)

        for i in range(0, gridSize, 2):
            self.window.columnconfigure(i, weight=1, minsize=10)
            self.window.rowconfigure(i, weight=1, minsize=10)
            for j in range(0, gridSize, 2):
                dot = tk.Label(bg="black", width=1, height=1)
                dot.grid(row=i, column=j, padx=5, pady=5)

            for j in range(1, gridSize, 2):
                button = tk.Button(bg=self.defaultButtonColor, height=1)
                button.grid(row=i, column=j, sticky='nsew', pady=5)
                button.bind("<Button-1>", self.ButtonClick)
                self.buttonsH[button] = (int(j / 2), int(i / 2))
                self.buttonsByMove[DotsAndBoxesMove(True, int(j / 2), int(i / 2), self.boardSize)] = button

        for i in range(1, gridSize, 2):
            for j in range(0, gridSize, 2):
                button = tk.Button(bg=self.defaultButtonColor, width=1)
                button.grid(row=i, column=j, sticky='nsew', padx=5)
                button.bind("<Button-1>", self.ButtonClick)
                self.buttonsV[button] = (int(j / 2), int(i / 2))
                self.buttonsByMove[DotsAndBoxesMove(False, int(j / 2), int(i / 2), self.boardSize)] = button

    def run(self):
        if self.aiIsTrue == self.db.GetCurrentPlayer():
            self.GetCpuMove()
        TkinterGui.run(self)

    def ButtonClick(self, event):
        button = event.widget
        if button['state'] == 'disabled':
            return
        if button in self.buttonsH:
            horizontal = True
            (x, y) = self.buttonsH[button]
        else:
            horizontal = False
            (x, y) = self.buttonsV[button]
        dbMove = DotsAndBoxesMove(horizontal, x, y, self.boardSize)
        if not self.db.IsValidMove(dbMove):
            return

        button["bg"] = "red"
        self.PlayMove(dbMove, "red")
        if self.db.GetCurrentPlayer() == self.aiIsTrue:
            self.GetCpuMove()

    def GetCpuMove(self):
        for b in self.buttonsByMove.values():
            b['state'] = "disabled"
            if b['bg'] == self.defaultButtonColor:
                b['bg'] = self.disabledButtonColor
        t = threading.Thread(target=self.ThreadedCpuMove, daemon=True)
        t.start()

    def ThreadedCpuMove(self):
        while not self.db.IsTerminal() and self.aiIsTrue == self.db.GetCurrentPlayer():
            (dbMove, _) = self.mcts.runSims(self.db)
            self.buttonsByMove[dbMove]["bg"] = "blue"
            print("CPU:" + str(dbMove))
            self.PlayMove(dbMove, "blue")

        for b in self.buttonsByMove.values():
            b['state'] = "normal"
            if b['bg'] == self.disabledButtonColor:
                b['bg'] = self.defaultButtonColor

    def PlayMove(self, dbMove, color):
        filledBoxes = self.db.PlayerMoveDb(dbMove)
        for b in filledBoxes:
            self.boxes[b[1]][b[0]]["bg"] = color

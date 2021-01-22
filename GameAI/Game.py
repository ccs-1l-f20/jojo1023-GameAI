from abc import ABC, abstractmethod

class Game(ABC):
    def __init__(self):
        self._turnCount = 0
        self._currentPlayer = True

    def GetCurrentPlayer(self):
        return self._currentPlayer

    @abstractmethod
    def PlayMove(self, move):
        pass

    @abstractmethod
    def IsTerminal(self):
        pass

    @abstractmethod
    def TerminalValue(self):
        pass

    @abstractmethod
    def IsValidMove(self, move):
        pass

class DotsAndBoxesMove:
    def __init__(self, horizontal, x, y):
        self.horizontal = horizontal
        self.x = x
        self.y = y

class DotsAndBoxes(Game):
    def __init__(self, size):
        Game.__init__(self)
        super().__init__()
        self._size = size
        self._tPoints = 0
        self._fPoints = 0
        self._hLines = [[False for i in range(size+1)] for j in range(size)]
        self._vLines = [[False for i in range(size)] for j in range(size + 1)]

    def IsValidMove(self, move):
        if move.horizontal:
            lines = self._hLines
        else:
            lines = self._vLines
        if move.x < len(lines):
            if move.y < len(lines[move.x]):
                return not lines[move.x][move.y]
        return False

    def _IsBoxFilled(self, x, y):
        if x >= self._size or y >= self._size:
            return False
        return self._hLines[x][y] and self._vLines[x][y] and self._hLines[x][y+1] and self._vLines[x + 1][y]

    def PlayMove(self, move):
        if move.horizontal:
            lines = self._hLines
        else:
            lines = self._vLines
        if lines[move.x][move.y]:
            return
        self._turnCount += 1
        lines[move.x][move.y] = True
        newPoints = self._IsBoxFilled(move.x, move.y)
        if move.horizontal:
            newPoints += self._IsBoxFilled(move.x, move.y - 1)
        else:
            newPoints += self._IsBoxFilled(move.x - 1, move.y)

        if newPoints > 0:
            if self._currentPlayer:
                self._tPoints += newPoints
            else:
                self._fPoints += newPoints
        else:
            self._currentPlayer = not self._currentPlayer

    def IsTerminal(self):
        # return self._turnCount >= (self._size * (self._size - 1))*2
        halfBoxCount = int((self._size * self._size)/2)
        return self._tPoints > halfBoxCount or self._fPoints > halfBoxCount

    def TerminalValue(self):
        if self._tPoints > self._fPoints:
            return 1.0
        elif self._fPoints > self._tPoints:
            return 0.0
        else:
            return 0.5

    def PlayerMove(self, pMove):
        def PlayerState(midGame):
            if midGame:
                print("Player: " + ("T" if self._currentPlayer else "F"))
            print("PointsT: " + str(self._tPoints))
            print("PointsF: " + str(self._fPoints) + "\n")
        if len(pMove) == 0:
            print("Invalid Move")
            PlayerState(True)
            return

        if pMove[0] == 'h':
            horizontal = True
        elif pMove[0] == 'v':
            horizontal = False
        else:
            print("Invalid Move")
            PlayerState(True)
            return

        splits = pMove[1:].split('x', 1)
        try:
            x = int(splits[0])
            y = int(splits[1])
        except ValueError:
            print("Invalid Move")
            PlayerState(True)
            return
        dbMove = DotsAndBoxesMove(horizontal, x, y)
        if not self.IsValidMove(dbMove):
            print("Invalid Move")
            PlayerState(True)
            return
        self.PlayMove(dbMove)
        if self.IsTerminal():
            terminalValue = self.TerminalValue()
            if terminalValue == 1:
                print("Game Over, T Wins!")
                PlayerState(False)
            elif terminalValue == 0:
                print("Game Over, F Wins!")
                PlayerState(False)
            else:
                print("Game Over, Draw")
                PlayerState(False)
        else:
            PlayerState(True)

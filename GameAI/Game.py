from abc import ABC, abstractmethod
from typing import List, Any, Tuple


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

    @abstractmethod
    def GetValidMoves(self):
        pass

    @abstractmethod
    def Clone(self):
        pass

    @abstractmethod
    def GetBoardState(self, player) -> List[float]:
        pass

    @abstractmethod
    def GetAllPossibleMoves(self):
        pass


class DotsAndBoxesMove:
    def __init__(self, horizontal, x, y, size):
        self.horizontal = horizontal
        self.x = x
        self.y = y
        self.size = size

    def __hash__(self):
        v = self.x * self.size + self.y
        if self.horizontal:
            return v
        else:
            return (self.size + 1) * self.size + v

    def __eq__(self, other):
        return (self.horizontal, self.x, self.y) == (other.horizontal, other.x, other.y)

    def __str__(self):
        return ("h" if self.horizontal else "v") + str(self.x) + "x" + str(self.y)

class DotsAndBoxes(Game):
    def __init__(self, size):
        super().__init__()
        self._size = size
        self._tPoints = 0
        self._fPoints = 0
        self._hLines = [[False for i in range(size + 1)] for j in range(size)]
        self._vLines = [[False for i in range(size)] for j in range(size + 1)]

    def IsValidMove(self, move):
        if move.x < 0 or move.y < 0:
            return False
        if move.horizontal:
            lines = self._hLines
        else:
            lines = self._vLines
        if move.x < len(lines):
            if move.y < len(lines[move.x]):
                return not lines[move.x][move.y]
        return False

    def GetValidMoves(self):
        moves = []
        for x in range(len(self._hLines)):
            for y in range(len(self._hLines[x])):
                if not self._hLines[x][y]:
                    moves.append(DotsAndBoxesMove(True, x, y, self._size))
        for x in range(len(self._vLines)):
            for y in range(len(self._vLines[x])):
                if not self._vLines[x][y]:
                    moves.append(DotsAndBoxesMove(False, x, y, self._size))
        return moves

    def GetAllPossibleMoves(self):
        moves = []
        for x in range(len(self._hLines)):
            for y in range(len(self._hLines[x])):
                moves.append(DotsAndBoxesMove(True, x, y, self._size))
        for x in range(len(self._vLines)):
            for y in range(len(self._vLines[x])):
                moves.append(DotsAndBoxesMove(False, x, y, self._size))
        return moves

    def _IsBoxFilled(self, x, y):
        if x >= self._size or y >= self._size or y < 0 or x < 0:
            return False
        return self._hLines[x][y] and self._vLines[x][y] and self._hLines[x][y + 1] and self._vLines[x + 1][y]

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
        halfBoxCount = int((self._size * self._size) / 2)
        return self._turnCount >= (self._size * (self._size + 1)) * 2 \
               or self._tPoints > halfBoxCount or self._fPoints > halfBoxCount

    def TerminalValue(self):
        if self._tPoints > self._fPoints:
            return 1.0
        elif self._fPoints > self._tPoints:
            return -1.0
        else:
            return 0.0

    def Clone(self):
        clone = DotsAndBoxes(self._size)
        clone._tPoints = self._tPoints
        clone._fPoints = self._fPoints
        clone._hLines = [row[:] for row in self._hLines]
        clone._vLines = [row[:] for row in self._vLines]
        clone._turnCount = self._turnCount
        clone._currentPlayer = self._currentPlayer
        return clone

    def _PlayerState(self, midGame):
        if midGame:
            print("Player: " + ("T" if self._currentPlayer else "F"))
        print("PointsT: " + str(self._tPoints))
        print("PointsF: " + str(self._fPoints) + "\n")

    def PlayerMoveDb(self, pMove: DotsAndBoxesMove):
        if not self.IsValidMove(pMove):
            print("Invalid Move")
            self._PlayerState(True)
            return
        self.PlayMove(pMove)

        filledBoxes = []
        if self._IsBoxFilled(pMove.x, pMove.y):
            filledBoxes.append((pMove.x, pMove.y))
        if pMove.horizontal and self._IsBoxFilled(pMove.x, pMove.y - 1):
            filledBoxes.append((pMove.x, pMove.y - 1))
        elif not pMove.horizontal and self._IsBoxFilled(pMove.x - 1, pMove.y):
            filledBoxes.append((pMove.x - 1, pMove.y))

        if self.IsTerminal():
            terminalValue = self.TerminalValue()
            if terminalValue == 1:
                print("Game Over, T Wins!")
                self._PlayerState(False)
            elif terminalValue == -1:
                print("Game Over, F Wins!")
                self._PlayerState(False)
            else:
                print("Game Over, Draw")
                self._PlayerState(False)
        else:
            self._PlayerState(True)
        return filledBoxes

    def PlayerMoveStr(self, pMove: str):
        if len(pMove) == 0:
            print("Invalid Move")
            self._PlayerState(True)
            return

        if pMove[0] == 'h':
            horizontal = True
        elif pMove[0] == 'v':
            horizontal = False
        else:
            print("Invalid Move")
            self._PlayerState(True)
            return

        splits = pMove[1:].split('x', 1)
        try:
            x = int(splits[0])
            y = int(splits[1])
        except ValueError:
            print("Invalid Move")
            self._PlayerState(True)
            return
        dbMove = DotsAndBoxesMove(horizontal, x, y, self._size)
        self.PlayerMoveDb(dbMove)

    def GetBoardState(self, player):
        state = []
        for i in range(0, len(self._hLines)):
            for line in self._hLines[i]:
                state.append(1 if line else 0)

        for i in range(0, len(self._vLines)):
            for line in self._vLines[i]:
                state.append(1 if line else 0)

        halfBoxCount = int((self._size * self._size) / 2)
        # if self._size % 2 == 0:
        #     halfBoxCount += 1

        if player:
            pPoints = self._tPoints
            oPoints = self._fPoints
        else:
            pPoints = self._fPoints
            oPoints = self._tPoints

        for i in range(0, halfBoxCount):
            state.append(1 if i == pPoints else 0)

        for i in range(0, halfBoxCount):
            state.append(1 if i == oPoints else 0)

        state.append(1 if player else 0)
        return state

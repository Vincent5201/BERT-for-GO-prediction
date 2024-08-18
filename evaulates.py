import numpy as np
from resnet_board import *
from math import tanh
from tools import transfer

def create_addboard(board, x, y, v, w):
    if v == 0 or x < 0 or y < 0 or x > 18 or y > 18:
        return
    if board[x][y] != 0:
        if w and board[x][y] >= v:
            return
        elif w == 0 and board[x][y] <= v:
            return
    if v != 6:
        if w:
            board[x][y] = v
        else:
            board[x][y] = -v
    create_addboard(board, x-1, y, v-1, w)
    create_addboard(board, x+1, y, v-1, w)
    create_addboard(board, x, y-1, v-1, w)
    create_addboard(board, x, y+1, v-1, w)

# evaluate black's win rate 
# not a normal way
def evaulate(games):
    game = [transfer(step) for step in games[0]]
    datas = np.zeros([1,4,19,19],  dtype=np.float32)
    for j, move in enumerate(game):
        x = int(move/19)
        y = int(move%19)
        Lchannel_01(datas, 0, x, y, j+1)
        Lchannel_3(datas, 0, x, y, j+1)
    Lchannel_2(datas, 0, len(games[0]))
    board = np.zeros([19,19])
    for i in range(19):
        for j in range(19):
            if datas[0][0][i][j]:
                addboard = np.zeros([19,19])
                create_addboard(addboard, i+1, j, 6, 0)
                board += addboard
            elif datas[0][1][i][j]:
                addboard = np.zeros([19,19])
                create_addboard(addboard, i+1, j, 6, 1)
                board += addboard
    for i in range(19):
        board[i][0] *= 2
        board[i][18] *= 2
    for j in range(1, 18):
        board[0][j] *= 2
        board[18][j] *= 2
    for i in range(1, 18):
        board[i][1] += (board[i][1]+1)//2
        board[i][17] += (board[i][17]+1)//2
    for j in range(2, 17):
        board[1][j] += (board[1][j]+1)//2
        board[17][j] += (board[17][j]+1)//2

    count = 0
    for i in range(19):
        for j in range(19):
            if datas[0][0][i][j]:
                if board[i][j] > 9:
                    count += 2
                elif board[i][j] > 6:
                    count += 1
            elif datas[0][1][i][j]:
                if board[i][j] < -9:
                    count -= 2
                elif board[i][j] < -6:
                    count -= 1
            else:
                if board[i][j] > 4:
                    count += 1
                elif board[i][j] < -4:
                    count -= 1
    ans = tanh((count-6.5)/5)
    ans += 1
    ans /= 2
    return ans * 100
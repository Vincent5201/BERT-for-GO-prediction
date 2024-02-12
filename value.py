import numpy as np

def spread_board(board, count, x, y):
    if count == 10:
        return board
    if x > 0 and (board[x-1][y] == 0 or board[x-1][y] > count):
        board[x-1][y] = count
        board = spread_board(board, count+1, x-1, y)
    if y > 0 and (board[x][y-1] == 0 or board[x][y-1] > count):
        board[x][y-1] = count
        board = spread_board(board, count+1, x, y-1)
    if x < 18 and (board[x+1][y] == 0 or board[x+1][y]  > count):
        board[x+1][y]  = count
        board = spread_board(board, count+1, x+1, y)
    if y < 18 and (board[x][y+1] == 0 or board[x][y+1]  > count):
        board[x][y+1]  = count
        board = spread_board(board, count+1, x, y+1)
    return board

def start_spread(board):
    for i in range(19):
        for j in range(19):
            if board[i][j] == 1:
                board = spread_board(board, 2, i, j)
    return board

def spread_value(game):
    white = game[0]
    black = game[1]
    white = start_spread(white)
    black = start_spread(black)
    score = 0
    for i in range(19):
        for j in range(19):
            if black[i][j] and white[i][j]:
                if black[i][j] < white[i][j]:
                    score += 1
                else:
                    socre -= 1
            elif black[i][j]:
                score += 1
            elif white[i][j]:
                score -= 1

    return score
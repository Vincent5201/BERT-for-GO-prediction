import numpy as np
from myDatasets import transfer, channel_01


def spread_board(board, count, x, y):
    if count == 7:
        return 
    if x > 0 and (board[x-1][y] == 0 or board[x-1][y] > count):
        board[x-1][y] = count
        spread_board(board, count+1, x-1, y)
    if y > 0 and (board[x][y-1] == 0 or board[x][y-1] > count):
        board[x][y-1] = count
        spread_board(board, count+1, x, y-1)
    if x < 18 and (board[x+1][y] == 0 or board[x+1][y]  > count):
        board[x+1][y]  = count
        spread_board(board, count+1, x+1, y)
    if y < 18 and (board[x][y+1] == 0 or board[x][y+1]  > count):
        board[x][y+1]  = count
        spread_board(board, count+1, x, y+1)
    return 

def start_spread(board):
    for i in range(19):
        for j in range(19):
            if board[i][j] == 1:
                spread_board(board, 2, i, j)
    return 

def spread_value(game):
    game = [transfer(step) for step in games]
    datas = np.zeros([1,2,19,19],  dtype=np.float32)
    for j, move in enumerate(game):
        x = int(move/19)
        y = int(move%19)
        channel_01(datas, 0, x, y, j)
    white = datas[0]
    black = datas[1]
    start_spread(white)
    start_spread(black)
    score = 0
    for i in range(19):
        for j in range(19):
            if black[i][j] and white[i][j]:
                if black[i][j] < white[i][j]:
                    score += 1
                else:
                    score -= 1
            elif black[i][j]:
                score += 1
            elif white[i][j]:
                score -= 1

    return score

if __name__ == "__main__":
    games = ["cd",'dq','pq','qd','oc','qo','co','ec', 'de','pe','np','fp']
    
    
    value = spread_value(games)
    print(value)
import numpy as np
from tqdm import tqdm

def channel_01(datas, k, x, y, turn):
    #plain1 is black
    #plain0 is white
    datas[k][turn%2][x][y] = 1
    live = set()
    died = set()
    def checkDie(x, y, p):
        ans = True
        pp = 0 if p else 1
        if (x, y, p) in live:
            return False
        if (x, y, p) in died:
            return True
        died.add((x, y, p))
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19:
                if datas[k][p][dx][dy] == 0 and datas[k][pp][dx][dy] == 0:
                    #neighbor is empty, alive
                    live.add((x, y, p))
                    return False
                if datas[k][p][dx][dy] == 1:
                    #neighbor is same, check neighbor is alive or not
                    #if one neighbor is alive, itself is alive 
                    ans = ans & checkDie(dx, dy, p)
        if ans:
            died.add((x, y, p))
        else:
            died.remove((x, y, p))
            live.add((x, y, p))
        return ans
    
    def del_die(x, y, p):
        datas[k][p][x][y] = 0
        for i in range(10,16):
            datas[k][i][x][y] = 0
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][p][dx][dy]:
                del_die(dx,dy,p)
        return
    
    directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for (dx, dy) in directions:
        if turn % 2:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][0][dx][dy]:
                if checkDie(dx, dy, 0):
                    del_die(dx, dy, 0)
        else:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][1][dx][dy]:
                if checkDie(dx, dy, 1):
                    del_die(dx, dy, 1)
    return

def channel_2(datas, k):
    # empty is 1
    datas[k][2] = np.logical_not(np.logical_or(datas[k][0],datas[k][1])).astype(int)
    return

def channel_3(datas, k, turn):
    #next turn (all 1/0)
    datas[k][3] = np.zeros([19,19]) if turn%2 else np.ones([19,19])
    return

def channel_49(datas, k, turn, labels):
    #last 4 moves
    turn = min(5, turn)
    p = 4
    kk = k-1
    for i in range(4,10):
        datas[k][i] = np.zeros([19,19])
    while turn >= 0:
        datas[k][p][int(labels[kk] / 19)][int(labels[kk] % 19)] = 1
        p += 1
        turn -= 1
        kk -= 1
    return

def channel_1015(datas, k, x, y, turn, mode="train",board=None):
    counted_empty = set()
    if mode == "board":
        idb = k
        k = 0
    def check_liberty(x, y, p):
        liberty = 0
        pp = 0 if p else 1
        datas[k][p][x][y] = 2
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19:
                if datas[k][pp][dx][dy] == 0 and datas[k][p][dx][dy] == 0:
                    if not (dx, dy) in counted_empty:
                        liberty += 1
                        counted_empty.add((dx,dy))
                elif datas[k][p][dx][dy] == 1:
                    liberty += check_liberty(dx, dy, p)
       
        datas[k][p][x][y] = 1        
        return liberty
    
    def set_liberty_plane(x, y, liberty):
        if mode == "train":
            if liberty < 6:
                for i in range(10,16):
                    if i == liberty+9:
                        datas[k][i][x][y] = 1
                    else:
                        datas[k][i][x][y] = 0
            else:
                for i in range(10,15):
                    datas[k][i][x][y] = 0
                datas[k][15][x][y] = 1
        else:
            board[idb][x][y] = min(6, liberty)
        return 
    
    def set_liberty(x, y, p, liberty):
        datas[k][p][x][y] = 2
        set_liberty_plane(x, y, liberty)
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][p][dx][dy] == 1:
                set_liberty(dx, dy, p, liberty)
        datas[k][p][x][y] = 1
        return
    
    if datas[k][2][x][y]:
        return
    
    ret = check_liberty(x, y, turn%2)
    set_liberty(x, y, turn%2, ret)
    pp = 0 if turn%2 else 1
    directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for (dx, dy) in directions:
        counted_empty.clear()
        if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][pp][dx][dy]:
            set_liberty(dx, dy, pp, check_liberty(dx, dy, pp))
    return 

def Lchannel_01(datas, k, x, y, turn):
    #plain1 is black
    #plain0 is white
    datas[k][turn%2][x][y] = 1
    live = set()
    died = set()
    def checkDie(x, y, p):
        ans = True
        pp = 0 if p else 1
        if (x, y, p) in live:
            return False
        if (x, y, p) in died:
            return True
        died.add((x, y, p))
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19:
                if datas[k][p][dx][dy] == 0 and datas[k][pp][dx][dy] == 0:
                    #neighbor is empty, alive
                    live.add((x, y, p))
                    return False
                if datas[k][p][dx][dy] == 1:
                    #neighbor is same, check neighbor is alive or not
                    #if one neighbor is alive, itself is alive 
                    ans = ans & checkDie(dx, dy, p)
        if ans:
            died.add((x, y, p))
        else:
            died.remove((x, y, p))
            live.add((x, y, p))
        return ans
    
    def del_die(x, y, p):
        datas[k][p][x][y] = 0
        datas[k][3][x][y] = 0
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][p][dx][dy]:
                del_die(dx,dy,p)
        return
    
    directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for (dx, dy) in directions:
        if turn % 2:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][0][dx][dy]:
                if checkDie(dx, dy, 0):
                    del_die(dx, dy, 0)
        else:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][1][dx][dy]:
                if checkDie(dx, dy, 1):
                    del_die(dx, dy, 1)
    return

def Lchannel_2(datas, k, turn):
    #next turn (all 1/0)
    datas[k][2] = np.zeros([19,19]) if turn%2 else np.ones([19,19])
    return

def Lchannel_3(datas, k, x, y, turn,):
    counted_empty = set()
    def check_liberty(x, y, p):
        liberty = 0
        pp = 0 if p else 1
        datas[k][p][x][y] = 2
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19:
                if datas[k][pp][dx][dy] == 0 and datas[k][p][dx][dy] == 0:
                    if not (dx, dy) in counted_empty:
                        liberty += 1
                        counted_empty.add((dx,dy))
                elif datas[k][p][dx][dy] == 1:
                    liberty += check_liberty(dx, dy, p)
       
        datas[k][p][x][y] = 1        
        return liberty
    
    def set_liberty_plane(x, y, liberty):
        datas[k][3][x][y] = min(6, liberty)
        return 
    
    def set_liberty(x, y, p, liberty):
        datas[k][p][x][y] = 2
        set_liberty_plane(x, y, liberty)
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][p][dx][dy] == 1:
                set_liberty(dx, dy, p, liberty)
        datas[k][p][x][y] = 1
        return
    
    if datas[k][0][x][y] == 0 and datas[k][1][x][y] == 0:
        return
    
    ret = check_liberty(x, y, turn%2)
    set_liberty(x, y, turn%2, ret)
    pp = 0 if turn%2 else 1
    directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    for (dx, dy) in directions:
        counted_empty.clear()
        if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][pp][dx][dy]:
            set_liberty(dx, dy, pp, check_liberty(dx, dy, pp))
    return 

def get_board(games):
    total_moves = 0
    for game in games:
        total_moves += len(game)
    if total_moves == 0:
        board = np.zeros((1, 19, 19))
        return board
    labels = np.zeros(total_moves)
    game_start = 0
    board = np.zeros((total_moves, 19, 19))
    datas = np.zeros([1,16,19,19],  dtype=np.float32)
    for _, game in tqdm(enumerate(games),total=len(games), leave=False):
        for j, move in enumerate(game):
            labels[game_start] = move
            if j == 0:
                datas = np.zeros([1,16,19,19],  dtype=np.float32)
            else:
                x = int(labels[game_start-1] // 19)
                y = int(labels[game_start-1] % 19)
                channel_01(datas, 0, x, y, j)
                channel_2(datas, 0)
                board[game_start] = board[game_start-1]
                channel_1015(datas, game_start, x, y, j, mode="board", board=board)
            game_start += 1

    return board
import numpy as np
import copy
from tqdm import tqdm
from math import sqrt, pow
import random

def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(n):
        matrix[i] = matrix[i][::-1]
    return matrix

def transformG(game, m):
    length = len(game)
    game = [m[int(move/19)][move%19] for move in game if move]
    length -= len(game)
    while(length > 0):
        game.append(0)
        length -= 1
    return game

def check_top_left(step):
    return step/19 < 10 and step%19 < 10

def check_top_right(step):
    return int(step/19) <= (step%19)

def extend(games):
    m0 = [[i * 19 + j for j in range(19)] for i in range(19)]
    mflip = np.transpose(np.array(copy.deepcopy(m0)))
    m90 = rotate(copy.deepcopy(m0))
    games90 = []
    games180 = []
    games270 = []
    gamesf = []
    for game in games:
        game90 = transformG(copy.deepcopy(game), m90)
        games90.append(game90)
        game180 = transformG(copy.deepcopy(game90), m90)
        games180.append(game180)
        game270 = transformG(copy.deepcopy(game180), m90)
        games270.append(game270)
        gamef = transformG(copy.deepcopy(game), mflip)
        gamesf.append(gamef)
        game90f = transformG(copy.deepcopy(game90), mflip)
        gamesf.append(game90f)
        game180f = transformG(copy.deepcopy(game180), mflip)
        gamesf.append(game180f)
        game270f = transformG(copy.deepcopy(game270), mflip)
        gamesf.append(game270f)
    games = np.concatenate((np.array(games),np.array(gamesf)), axis=0)
    return games

def check(game, data_source, num_moves):
    first_steps = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp","cc", "cq", "qc","qq"]
    if len(game) < num_moves:
        return False
    for i, step in enumerate(game):
        if isinstance(step, float):
            return True
        if data_source == 'foxwq':
            if i == 0:
                if step != 'B' and step != 'W':
                    return False
            elif i == 1:
                if not (step in first_steps):
                    return False 
            else:
                if(len(step) != 2 or step[0]<'a' or step[0]>'s' or step[1]<'a' or step[1]>'s'):
                    return False
        elif data_source == 'pros':
            if i == 0:
                if not (step in first_steps):
                    return False
            else:
                if(len(step) != 2 or step[0]<'a' or step[0]>'s' or step[1]<'a' or step[1]>'s'):
                    return False
        else:
            print(f'skip_check_{data_source}')
    return True

def transfer(step):
    if isinstance(step, float):
       return 0
    return (ord(step[0])-97)*19 + (ord(step[1])-97) 

def transfer_back(step):
    return chr(int(step/19)+97)+chr(int(step%19)+97) 

def stepbystep(game, shift=0):
    num_moves = len(game)
    rgames = [[game[j]+shift if j <= i else 0 for j in range(num_moves)] for i in range(num_moves)]
    return rgames

def get_tensor_memory_size(tensor):
    return tensor.numel() * tensor.element_size()

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

def myaccn(pred, true, n):
    total = len(true)
    correct = 0
    for i, p in tqdm(enumerate(pred), total=len(pred), leave=False):
        sorted_indices = (-p).argsort()
        top_k_indices = sorted_indices[:n]  
        if true[i] in top_k_indices:
            correct += 1
    return correct / total

def myaccn_split(pred, true, n, split, num_move):
    correct = [0]*split
    for i, p in tqdm(enumerate(pred), total=len(pred), leave=False):
        sorted_indices = (-p).argsort()
        top_k_indices = sorted_indices[:n]  
        if true[i] in top_k_indices:
            correct[int((i%num_move)/int(num_move/split))] += 1
    part_total = len(true)/split
    for i in range(split):
        correct[i] /= part_total
    return correct 

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


def distance(m1, m2):
    if m1 == m2:
        return 0
    return sqrt(pow(m2//19-m1//19,2)+ pow(m2%19-m1%19,2))

def shuffle_intervals(game, battle_break, pos):
    intervals = battle_break[:pos]
    ranges = [(intervals[i], intervals[i+1]) for i in range(len(intervals) - 1)]
    np.random.shuffle(ranges)
    shuffled_game = np.concatenate([game[start:end] for start, end in ranges])
    game[intervals[0]:intervals[-1]] = shuffled_game
    return game

def shuffle_battle(games, battle_break):
    count = 0
    pos = 0
    shuffle_games = []
    if len(battle_break) < 3:
        return games, 0
    for i, game in enumerate(games):
        if pos < len(battle_break):
            if i == battle_break[pos]:
                pos += 1
                if pos > 2 and random.randint(0,1):
                    shuffle_games.append(shuffle_intervals(game, battle_break, pos))
                    count += 1
        else:
            break
    return shuffle_games, count
import numpy as np
import copy
from tqdm import tqdm
from math import sqrt, pow

# rotate 90 degree
def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(n):
        matrix[i] = matrix[i][::-1]
    return matrix

# transfer bu map m
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

# rotate and flip board
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

# check input data
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

# accuracy
def myaccn(pred, true, n):
    total = len(true)
    correct = 0
    for i, p in tqdm(enumerate(pred), total=len(pred), leave=False):
        sorted_indices = (-p).argsort()
        top_k_indices = sorted_indices[:n]  
        if true[i] in top_k_indices:
            correct += 1
    return correct / total

# accuracy in every length of games
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

def distance(m1, m2):
    if m1 == m2:
        return 0
    return sqrt(pow(m2//19-m1//19,2)+ pow(m2%19-m1%19,2))

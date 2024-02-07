import numpy as np
import pandas as pd
import copy
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for i in range(n):
        matrix[i] = matrix[i][::-1]
    return matrix

def transformG(game, m):
    game = [m[int(move/19)][move%19] for move in game]
    return game

def check_top_left(step):
    return step/19 < 10 and step%19 < 10

def check_top_right(step):
    return int(step/19) <= (step%19)

def top_left(games):
    m0 = [[i * 19 + j for j in range(19)] for i in range(19)]
    mflip = np.transpose(np.array(copy.deepcopy(m0)))
    m90 = rotate(copy.deepcopy(m0))
    lgames = []
    for game in games:
        if check_top_left(game[0]):
            if(check_top_right(game[1])):
                lgames.append(game)
            else:
                lgames.append(transformG(game, mflip))
            continue
        game = transformG(game, m90)
        if check_top_left(game[0]):
            if(check_top_right(game[1])):
                lgames.append(game)
            else:
                lgames.append(transformG(game, mflip))
            continue
        game = transformG(game, m90)
        if check_top_left(game[0]):
            if(check_top_right(game[1])):
                lgames.append(game)
            else:
                lgames.append(transformG(game, mflip))
            continue
        game = transformG(game, m90)
        if check_top_left(game[0]):
            if(check_top_right(game[1])):
                lgames.append(game)
            else:
                lgames.append(transformG(game, mflip))
            continue
    return lgames

def check(game, data_source):
    first_steps = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp","cc", "cq", "qc","qq"]
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

def stepbystep(game):
    num_moves = len(game)
    rgames = [[game[j] if j <= i else 0 for j in range(num_moves)] for i in range(num_moves)]
    return rgames

def get_tensor_memory_size(tensor):
    numel = tensor.numel()
    element_size = tensor.element_size()
    total_memory_size = numel * element_size
    return total_memory_size

def channel_01(datas, k, x, y, turn):
    #plain1 is black
    #plain0 is white
    datas[k][turn%2][x][y] = 1
    #del die 
    def checkDie(x, y, p):
        ans = True
        pp = 1
        if p:
            pp = 0
        if x > 0:
            if datas[k][p][x-1][y] == 0 and datas[k][pp][x-1][y] == 0:
                #neighbor is empty, alive
                return False
            if datas[k][p][x-1][y] == 1:
                #neighbor is same, check neighbor is alive or not
                datas[k][p][x][y] = 2
                #if one neighbor is alive, itself is alive 
                ans = ans & checkDie(x-1, y, p)
                datas[k][p][x][y] = 1
        if y > 0:
            if datas[k][p][x][y-1] == 0 and datas[k][pp][x][y-1] == 0:
                return False
            if datas[k][p][x][y-1] == 1:
                datas[k][p][x][y] = 2
                ans = ans & checkDie(x, y-1, p)
                datas[k][p][x][y] = 1
        if x < 18:
            if datas[k][p][x+1][y] == 0 and datas[k][pp][x+1][y] == 0:
                return False
            if datas[k][p][x+1][y] == 1:
                datas[k][p][x][y] = 2
                ans = ans & checkDie(x+1, y, p)
                datas[k][p][x][y] = 1
        if y < 18:
            if datas[k][p][x][y+1] == 0 and datas[k][pp][x][y+1] == 0:
                return False
            if datas[k][p][x][y+1] == 1:
                datas[k][p][x][y] = 2
                ans = ans & checkDie(x, y+1, p)
                datas[k][p][x][y] = 1
        if ans:
            # if die, delete it
            datas[k][p][x][y] = 0
        return ans
    
    if turn % 2:
        if x > 0 and datas[k][0][x-1][y]:
            checkDie(x-1, y, 0)
        if y > 0 and datas[k][0][x][y-1]:
            checkDie(x, y-1, 0)
        if x < 18 and datas[k][0][x+1][y]:
            checkDie(x+1, y, 0)
        if y < 18 and datas[k][0][x][y+1]:
            checkDie(x, y+1, 0)
    else:
        if x > 0 and datas[k][1][x-1][y]:
            checkDie(x-1, y, 1)
        if y > 0 and datas[k][1][x][y-1]:
            checkDie(x, y-1, 1)
        if x < 18 and datas[k][1][x+1][y]:
            checkDie(x+1, y, 1)
        if y < 18 and datas[k][1][x][y+1]:
            checkDie(x, y+1, 1)
    return

def channel_3(datas, k, turn):
    #next turn (all 1/0)
    if turn % 2 == 0:
        datas[k][3] = np.ones([19,19])
    else:
        datas[k][3] = np.zeros([19,19])
    return

def channel_49(datas, k, turn, labels):
    #last 4 moves
    turn = min(5, turn)
    p = 4
    kk = k-1
    datas[k][4] = np.zeros([19,19])
    datas[k][5] = np.zeros([19,19])
    datas[k][6] = np.zeros([19,19])
    datas[k][7] = np.zeros([19,19])
    datas[k][8] = np.zeros([19,19])
    datas[k][9] = np.zeros([19,19])
    while turn >= 0:
        datas[k][p][int(labels[kk] / 19)][int(labels[kk] % 19)] = 1
        p += 1
        turn -= 1
        kk -= 1
    return

def channel_1015(datas, k, x, y, turn):
    def set_liberty(k, x, y, lives):
        for live in range(10, 16):
            if live == lives:
                datas[k][live][x][y] = 1
            else:
                datas[k][live][x][y] = 0
    #check liberty
    def checklive(x, y, p):
        lives = 0
        pp = 1
        if p:
            pp = 0
        if x > 0:
            if datas[k][p][x-1][y] == 0 and datas[k][pp][x-1][y] == 0:
                lives += 1
                datas[k][p][x-1][y] = 2
            if datas[k][p][x-1][y] == 1:
                #neighbor is same, check neighbor
                datas[k][p][x][y] = 2
                lives += checklive(x-1, y, p)
                datas[k][p][x][y] = 1
            if datas[k][p][x-1][y] == 2:
                datas[k][p][x-1][y] = 0
        if y > 0:
            if datas[k][p][x][y-1] == 0 and datas[k][pp][x][y-1] == 0:
                lives += 1
                datas[k][p][x][y-1] = 2
            if datas[k][p][x][y-1] == 1:
                datas[k][p][x][y] = 2
                lives += checklive(x, y-1, p)
                datas[k][p][x][y] = 1
            if datas[k][p][x][y-1] == 2:
                datas[k][p][x][y-1] = 0
        if x < 18:
            if datas[k][p][x+1][y] == 0 and datas[k][pp][x+1][y] == 0:
                lives += 1
                datas[k][p][x+1][y] = 2
            if datas[k][p][x+1][y] == 1:
                datas[k][p][x][y] = 2
                lives += checklive(x+1, y, p)
                datas[k][p][x][y] = 1
            if datas[k][p][x+1][y] == 2:
                datas[k][p][x+1][y] = 0
        if y < 18:
            if datas[k][p][x][y+1] == 0 and datas[k][pp][x][y+1] == 0:
                lives += 1
                datas[k][p][x][y+1] = 2
            if datas[k][p][x][y+1] == 1:
                datas[k][p][x][y] = 2
                lives += checklive(x, y+1, p)
                datas[k][p][x][y] = 1
            if datas[k][p][x][y+1] == 2:
                datas[k][p][x][y+1] = 0
        
        if lives < 6:
            set_liberty(k, x, y, lives+9)
        else:
            set_liberty(k, x, y, 15)
        return lives
    
    if turn % 2:
        checklive(x, y, 1)
        if x > 0 and datas[k][0][x-1][y]:
            checklive(x-1, y, 0)
        if y > 0 and datas[k][0][x][y-1]:
            checklive(x, y-1, 0)
        if x < 18 and datas[k][0][x+1][y]:
            checklive(x+1, y, 0)
        if y < 18 and datas[k][0][x][y+1]:
            checklive(x, y+1, 0)
    else:
        checklive(x, y, 0)
        if x > 0 and datas[k][1][x-1][y]:
            checklive(x-1, y, 1)
        if y > 0 and datas[k][1][x][y-1]:
            checklive(x, y-1, 1)
        if x < 18 and datas[k][1][x+1][y]:
            checklive(x+1, y, 1)
        if y < 18 and datas[k][1][x][y+1]:
            checklive(x, y+1, 1)
    return

def channel_2(datas, k):
    # empty is 1
    datas[k][2] = np.logical_not(np.logical_or(datas[k][0],datas[k][1])).astype(int)
    return

class PicturesDataset(Dataset):
    # data loading
    def __init__(self,games, num_moves):
        moves_num = []
        for game in games:
            moves_num.append(len(game))
        total_moves = np.sum(moves_num)
        datas = np.zeros([total_moves,16,19,19],  dtype=np.float32)
        labels = np.zeros(total_moves)

        game_start = 0
        for i, game in tqdm(enumerate(games),total=len(games), leave=False):
            for j, move in enumerate(game):
                labels[game_start] = move
                if j == 0:
                    datas[game_start][2] = np.ones([19,19])
                    datas[game_start][3] = np.ones([19,19])
                else:
                    x = int(labels[game_start-1] / 19)
                    y = int(labels[game_start-1] % 19)
                    datas[game_start] = datas[game_start-1]
                    channel_01(datas, game_start, x, y, j)
                    channel_2(datas, game_start)
                    channel_3(datas, game_start, j)
                    channel_49(datas, game_start, j, labels)
                    channel_1015(datas, game_start, x, y, j)
                game_start += 1
        
        self.x = torch.tensor(datas)
        self.y = torch.tensor(labels).long()
        self.n_samples = total_moves
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class WordsDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves):
        gamesall = []
        for game in tqdm(games, total = len(games), leave=False):
            result = stepbystep(game)
            gamesall.append(result)
        gamesall = np.array(gamesall)
        gamesall = gamesall.reshape(gamesall.shape[0]*gamesall.shape[1],gamesall.shape[2]) 
        print("steps finish")
        gamesall = np.unique(gamesall, axis=0)
        print("unique finish")

        total_steps = gamesall.shape[0]
        y = [0]*(total_steps)
        for i in tqdm(range(total_steps), total=total_steps, leave=False):
            last = 0
            while(last < num_moves and gamesall[i][last] != 0):
                gamesall[i][last] += 1
                last += 1
            last -= 1
            y[i] = gamesall[i][last]-1
            gamesall[i][last] = 0
        print("data finish")

        self.x = torch.tensor(gamesall).long()
        self.y = (torch.tensor(y)).long()
        self.mask = (self.x != 0).detach().long()
        self.n_samples = total_steps
        
    def __getitem__(self, index):  
        return self.x[index], self.mask[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
def get_datasets(path, data_type, data_source, data_size, num_moves, split_rate, be_top_left):
    df = pd.read_csv(path, encoding="ISO-8859-1", on_bad_lines='skip').head(data_size)
    df = df.sample(frac=1,replace=False).reset_index(drop=True).to_numpy()
    before_chcek = len(df)
    games = [game for game in df if check(game, data_source)]
    after_check = len(games)
    print(f'check_rate:{after_check/before_chcek}')
    print(f'has {after_check} games')
    games = np.array(games)
    if data_source == "foxwq":
        games = np.delete(games, 0, axis=1)
    games = [[transfer(step) for step in game[:num_moves]] for game in games]
    print("transfer finish")
    if be_top_left:
        games = top_left(games)
    split = int(after_check * split_rate)
    games_train = games[split:]        
    games_eval = games[:split]
    if data_type == 'Word':
        train_dataset = WordsDataset(games_train,  num_moves)
        eval_dataset = WordsDataset(games_eval,  num_moves)
    elif data_type == 'Picture':
        train_dataset = PicturesDataset(games_train, num_moves)
        eval_dataset = PicturesDataset(games_eval, num_moves)
    print(f'trainData shape:{train_dataset.x.shape}')
    print(f'trainData memory size:{get_tensor_memory_size(train_dataset.x)}')
    print(f'evalData shape:{eval_dataset.x.shape}')
    print(f'evalData memory size:{get_tensor_memory_size(eval_dataset.x)}')
    return train_dataset, eval_dataset


if __name__ == "__main__":
    path = 'D:\codes\python\.vscode\Transformer_Go\datas\data_Foxwq_9d.csv'
    data_source = "foxwq"
    data_type = 'Picture'
    num_moves = 80
    data_size = 10
    split_rate = 0.1
    be_top_left = False
    trainData, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate, be_top_left)
    print(trainData.x[2][0])
    print(trainData.x[2][1])


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

def extend(games):
    m0 = [[i * 19 + j for j in range(19)] for i in range(19)]
    mflip = np.transpose(np.array(copy.deepcopy(m0)))
    m90 = rotate(copy.deepcopy(m0))
    games90 = []
    games180 = []
    games270 = []
    for game in games:
        game90 = copy.deepcopy(game)
        game90 = transformG(game90, m90)
        games90.append(game90)
        game180 = copy.deepcopy(game90)
        game180 = transformG(game180, m90)
        games180.append(game180)
        game270 = copy.deepcopy(game180)
        game270 = transformG(game270, m90)
        games270.append(game270)
    
    games = np.concatenate((np.array(games),np.array(games90), np.array(games180), np.array(games270)), axis=0)
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

def stepbystep(game, min_move=None, max_move=None):
    num_moves = len(game)
    if min_move is None and max_move is None:
        rgames = [[game[j] if j <= i else 0 for j in range(num_moves)] for i in range(num_moves)]
    elif min_move is None:
        rgames = [[game[j] if j <= i else 0 for j in range(num_moves)] for i in range(max_move)]
    elif max_move is None:
        rgames = [[game[j] if j <= i else 0 for j in range(num_moves)] for i in range(min_move, num_moves)]
    else:
        rgames = [[game[j] if j <= i else 0 for j in range(num_moves)] for i in range(min_move, max_move)]
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
            datas[k][15][x][y] = 0
            datas[k][14][x][y] = 0
            datas[k][13][x][y] = 0
            datas[k][12][x][y] = 0
            datas[k][11][x][y] = 0
            datas[k][10][x][y] = 0
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

    def check_liberty(datas, k, x, y, p):
        pp = 1
        liberty = 0
        if p:
            pp = 0
        datas[k][p][x][y] = 2
        if x > 0:
            if datas[k][pp][x-1][y] == 0 and datas[k][p][x-1][y] == 0:
                liberty += 1
                datas[k][p][x-1][y] = 3
            elif datas[k][p][x-1][y] == 1:
                liberty += check_liberty(datas, k, x-1, y, p)
        if y > 0:
            if datas[k][pp][x][y-1] == 0 and datas[k][p][x][y-1] == 0:
                liberty += 1
                datas[k][p][x][y-1] = 3
            elif datas[k][p][x][y-1] == 1:
                liberty += check_liberty(datas, k, x, y-1, p)
        if x < 18:
            if datas[k][pp][x+1][y] == 0 and datas[k][p][x+1][y] == 0:
                liberty += 1
                datas[k][p][x+1][y] = 3
            elif datas[k][p][x+1][y] == 1:
                liberty += check_liberty(datas, k, x+1, y, p)
        if y < 18:
            if datas[k][pp][x][y+1] == 0 and datas[k][p][x][y+1] == 0:
                liberty += 1
                datas[k][p][x][y+1] = 3
            elif datas[k][p][x][y+1] == 1:
                liberty += check_liberty(datas, k, x, y+1, p)
        datas[k][p][x][y] = 1
        if x > 0 and datas[k][p][x-1][y] == 3:
            datas[k][p][x-1][y] = 0
        if y > 0 and datas[k][p][x][y-1] == 3:
            datas[k][p][x][y-1] = 0
        if x < 18 and datas[k][p][x+1][y] == 3:
           datas[k][p][x+1][y] = 0
        if y < 18 and datas[k][p][x][y+1] == 3:
            datas[k][p][x][y+1] = 0
        return liberty
    
    def set_liberty_plane(datas, k, x, y, liberty):
        if liberty < 6:
            for i in range(10,16):
                if i == liberty+9:
                    datas[k][i][x][y] = 1
                else:
                    datas[k][i][x][y] = 0
        else:
            datas[k][15][x][y] = 1
            datas[k][14][x][y] = 0
            datas[k][13][x][y] = 0
            datas[k][12][x][y] = 0
            datas[k][11][x][y] = 0
            datas[k][10][x][y] = 0
        return 
    
    
    def set_liberty(datas, k, x, y, p, liberty):
        datas[k][p][x][y] = 2
        set_liberty_plane(datas, k, x, y, liberty)
        if x > 0 and datas[k][p][x-1][y] == 1:
            set_liberty(datas, k, x-1, y, p, liberty)
        if y > 0 and datas[k][p][x][y-1] == 1:
            set_liberty(datas, k, x, y-1, p, liberty)
        if x < 18 and datas[k][p][x+1][y] == 1:
            set_liberty(datas, k, x+1, y, p, liberty)
        if y < 18 and datas[k][p][x][y+1] == 1:
            set_liberty(datas, k, x, y+1, p, liberty)
        datas[k][p][x][y] = 1
        return
    if datas[k][2][x][y]:
        ret = 0
    else:
        ret = check_liberty(datas, k, x, y, turn%2)
        set_liberty(datas, k, x, y, turn%2, ret)
    pp = 1
    if turn%2:
        pp = 0
    if x > 0 and datas[k][pp][x-1][y]:
        ret1 = check_liberty(datas, k, x-1, y, pp)
        set_liberty(datas, k, x-1, y, pp, ret1)
    if y > 0 and datas[k][pp][x][y-1]:
        ret2 = check_liberty(datas, k, x, y-1, pp)
        set_liberty(datas, k, x, y-1, pp, ret2)
    if x < 18 and datas[k][pp][x+1][y]:
        ret3 = check_liberty(datas, k, x+1, y, pp)
        set_liberty(datas, k, x+1, y, pp, ret3)
    if y < 18 and datas[k][pp][x][y+1]:
        ret4 = check_liberty(datas, k, x, y+1, pp)
        set_liberty(datas, k, x, y+1, pp, ret4)

    return ret

def channel_2(datas, k):
    # empty is 1
    datas[k][2] = np.logical_not(np.logical_or(datas[k][0],datas[k][1])).astype(int)
    return


def find_joseki(games):
    joseki_dict = {}
    games = games[:, :, :10, :10]
    for game in tqdm(games, total=len(games)):
        nonzero_count0 = np.count_nonzero(game[0])
        nonzero_count1 = np.count_nonzero(game[1])
        if nonzero_count0 + nonzero_count1 > 2:
            key = tuple(game.reshape(200))
            try:
                joseki_dict[key] += 1
            except:
                joseki_dict[key] = 1
            sorted_d = dict(sorted(joseki_dict.items(), key=lambda item: item[1], reverse=True))
            joseki_dict = dict(list(sorted_d.items())[:1000])
    print(joseki_dict.values())

class PicturesDataset(Dataset):
    # data loading
    def __init__(self,games, num_moves, mim_move=None, max_move = None):
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
        #find_joseki(datas[:,:2,:,:])
        self.x = torch.tensor(datas)
        self.y = torch.tensor(labels).long()
        self.n_samples = total_moves
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class WordsDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves, mim_move=None, max_move=None, train=True):
        gamesall = []
        for game in tqdm(games, total = len(games), leave=False):
            result = stepbystep(game, mim_move, max_move)
            gamesall.append(result)
        gamesall = np.array(gamesall)
        gamesall = gamesall.reshape(gamesall.shape[0]*gamesall.shape[1],gamesall.shape[2]) 
        print("steps finish")
        if train:
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
            gamesall[i][last] = 362
        print("data finish")
        gamesall = np.array(gamesall)
        gamesall = np.insert(gamesall, 0, 363, axis=1)

        self.x = torch.tensor(gamesall).long()
        self.y = (torch.tensor(y)).long()
        self.mask = (self.x != 0).detach().long()
        self.n_samples = total_steps
        
    def __getitem__(self, index):  
        return self.x[index], self.mask[index], self.y[index]

    def __len__(self):
        return self.n_samples

class BERTPretrainDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves):
        #next_sentence data
        
        def deal_data(games, length, num_moves):
            half = int(length/2)
            games = np.array(games)

            mask = torch.rand([games.shape[0]]) < 0.5
            games_1 = games[mask]
            games_0 = games[~mask]
            games_a, games_b = np.hsplit(games_1, [half])
            np.random.shuffle(games_b)
            games_1 = np.concatenate((np.insert(games_a, half, 362, axis=1), np.insert(games_b, length-half, 362, axis=1)), axis=1)

            games_0 = np.insert(games_0, half, 362, axis=1)
            games_0 = np.insert(games_0, length+1, 362, axis=1)

            games = np.insert(np.concatenate((games_1, games_0), axis=0), 0, 363, axis=1)
            next_sentence_labels = np.concatenate((torch.ones([games_1.shape[0]]), torch.zeros([games_0.shape[0]])), axis=0)

            # 15% mask data
            labels = copy.deepcopy(games)
            mask = (torch.rand(games.shape) < 0.15) * (games != 0) * (games != 362) * (games != 363)
            for i in range(games.shape[0]):
                games[i, torch.flatten(mask[i].nonzero()).tolist()] = 364

            token_type = np.concatenate((torch.zeros([games.shape[0], half+2]), torch.ones([games.shape[0], num_moves+1-half])), axis=1)
            return games, labels, token_type, next_sentence_labels

        games_record = np.zeros([len(games),num_moves])

        gamesall = []
        labels = []
        token_types = []
        next_sentence_labels = []
     
        for i in range(len(games)):
            games_record[i][0] = games[i][0]
        for i in tqdm(range(1, num_moves), total=num_moves-1):
            for j in range(len(games_record)):
                games_record[j][i] = games[j][i]
            games_tmp, label_tmp, token_type_tmp, next_sentence_labels_tmp = deal_data(copy.deepcopy(games_record), i+1, num_moves)
            
            gamesall.append(games_tmp)
            labels.append(label_tmp)
            token_types.append(token_type_tmp)
            next_sentence_labels.append(next_sentence_labels_tmp)
      
        gamesall = np.array(gamesall).reshape((num_moves-1)*len(games), num_moves+3)
        labels = np.array(labels).reshape((num_moves-1)*len(games), num_moves+3)
        token_types = np.array(token_types).reshape((num_moves-1)*len(games), num_moves+3)
        next_sentence_labels = np.array(next_sentence_labels).reshape((num_moves-1)*len(games))

        self.x = torch.tensor(gamesall).long()
        self.y = torch.tensor(labels).long()
        self.mask = (self.x != 0).detach().long()
        self.token_type = torch.tensor(token_types).long()
        self.next_sentence_labels = torch.tensor(next_sentence_labels).long()
        self.n_samples = self.x.shape[0]


    def __getitem__(self, index):  
        return self.x[index], self.mask[index], self.token_type[index], self.next_sentence_labels[index], self.y[index]

    def __len__(self):
        return self.n_samples

def get_datasets(path, data_type, data_source, data_size, num_moves, split_rate, be_top_left, train=True, min_move=None, max_move=None):
    df = pd.read_csv(path, encoding="ISO-8859-1", on_bad_lines='skip').head(data_size)
    df = df.sample(frac=1,replace=False,random_state=17).reset_index(drop=True).to_numpy()
    before_chcek = len(df)
    games = [game for game in df if check(game, data_source, num_moves)]
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
    train_dataset = None
    eval_dataset = None
    if data_type == 'Word':
        if train:
            train_dataset = WordsDataset(games[split:],  num_moves)
        eval_dataset = WordsDataset(games[:split],  num_moves, min_move, max_move, train=train)
    elif data_type == 'Picture':
        if train:
            train_dataset = PicturesDataset(games[split:], num_moves)
        eval_dataset = PicturesDataset(games[:split], num_moves, min_move, max_move)
    elif data_type == "Pretrain":
        games = extend(games)
        train_dataset = BERTPretrainDataset(games, num_moves)
        eval_dataset = None
    if not train_dataset is None:
        print(f'trainData shape:{train_dataset.x.shape}')
        print(f'trainData memory size:{get_tensor_memory_size(train_dataset.x)}')
    if not eval_dataset is None:
        print(f'evalData shape:{eval_dataset.x.shape}')
        print(f'evalData memory size:{get_tensor_memory_size(eval_dataset.x)}')
    return train_dataset, eval_dataset


if __name__ == "__main__":
    path = 'datas/data_240119.csv'

    data_source = "pros"
    data_type = 'Picture'
    num_moves = 80
    data_size = 30000
    split_rate = 0.1
    be_top_left = False
    trainData, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate, be_top_left)
  
    print(trainData.n_samples)



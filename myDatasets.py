import numpy as np
import pandas as pd
import copy
import random
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
        game90 = transformG(copy.deepcopy(game), m90)
        games90.append(game90)
        game180 = transformG(copy.deepcopy(game90), m90)
        games180.append(game180)
        game270 = transformG(copy.deepcopy(game180), m90)
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

def stepbystep(game):
    num_moves = len(game)
    rgames = [[game[j]+1 if j <= i else 0 for j in range(num_moves)] for i in range(num_moves)]
    return rgames

def get_tensor_memory_size(tensor):
    return tensor.numel() * tensor.element_size()

def shuffle_pos(games):
    mat = [345, 160, 143, 207, 257, 2, 350, 309, 88, 346, 255, 282, 180, 275, 171, 115, 23, 79, 324, 343, 231, 227, 9, 228,
    140, 185, 85, 240, 123, 37, 203, 223, 4, 339, 243, 261, 103, 214, 209, 31, 182, 359, 89, 3, 111, 245, 117, 278, 310,
    70, 330, 307, 148, 306, 217, 69, 54, 100, 64, 82, 284, 74, 179, 329, 186, 105, 222, 201, 220, 305, 41, 297, 76,
    136, 328, 1, 250, 272, 157, 314, 14, 43, 126, 164, 58, 151, 17, 145, 249, 28, 291, 132, 169, 83, 113, 91, 267, 335,
    340, 286, 78, 277, 127, 322, 276, 273, 61, 218, 56, 172, 49, 73, 230, 139, 87, 264, 141, 104, 102, 355, 344, 239,
    313, 176, 51, 259, 106, 236, 39, 352, 177, 166, 29, 338, 241, 337, 81, 327, 146, 129, 22, 165, 260, 281, 234, 158,
    348, 118, 45, 349, 137, 194, 25, 190, 110, 130, 20, 191, 246, 15, 142, 175, 316, 265, 33, 356, 149, 315, 155,
    12, 212, 296, 200, 162, 319, 262, 325, 107, 251, 221, 342, 173, 202, 163, 320, 71, 188, 235, 96, 210, 233, 119, 279,
    174, 333, 92, 68, 292, 323, 244, 247, 204, 13, 248, 192, 354, 30, 287, 99, 147, 258, 205, 304, 332, 229, 303, 122,
    150, 288, 131, 124, 5, 6, 59, 52, 311, 318, 11, 271, 270, 336, 55, 232, 295, 269, 18, 199, 34, 213, 114, 42, 302,
    21, 167, 16, 98, 40, 153, 152, 211, 46, 357, 134, 27, 312, 67, 300, 256, 48, 156, 219, 326, 215, 268, 80, 274,
    195, 263, 77, 154, 35, 63, 86, 144, 84, 44, 159, 242, 301, 72, 38, 125, 331, 317, 120, 112, 196, 65, 293, 47, 237,
    8, 347, 108, 128, 116, 75, 294, 10, 62, 183, 24, 351, 181, 101, 224, 238, 341, 198, 93, 353, 280, 358, 36, 285,
    121, 97, 170, 94, 321, 178, 184, 0, 193, 289, 66, 283, 298, 19, 138, 90, 60, 334, 252, 50, 225, 53, 253, 168, 290,
    254, 266, 189, 7, 57, 206, 308, 197, 32, 133, 135, 187, 161, 26, 226, 299, 109, 95, 216, 208, 360]

    for i, game in enumerate(games):
        for j, move in enumerate(game):
            game[j] = mat[move]
        games[i] = game
    return games

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

def channel_1015(datas, k, x, y, turn):
    def check_liberty(x, y, p):
        liberty = 0
        pp = 0 if p else 1
        datas[k][p][x][y] = 2
        directions = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
        for (dx, dy) in directions:
            if dx >= 0 and dx < 19 and dy >= 0 and dy < 19:
                if datas[k][pp][dx][dy] == 0 and datas[k][p][dx][dy] == 0:
                    liberty += 1
                    datas[k][p][dx][dy] = 3
                elif datas[k][p][dx][dy] == 1:
                    liberty += check_liberty(dx, dy, p)
       
        datas[k][p][x][y] = 1        
        return liberty
    
    def set_liberty_plane(x, y, liberty):
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
        if dx >= 0 and dx < 19 and dy >= 0 and dy < 19 and datas[k][pp][dx][dy]:
            set_liberty(dx, dy, pp, check_liberty(dx, dy, pp))
        for i in range(19):
            for j in range(19):
                if datas[k][0][i][j] == 3:
                    datas[k][0][i][j] = 0
                if datas[k][1][i][j] == 3:
                    datas[k][1][i][j] = 0

    return ret


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
    def __init__(self,games, num_moves):
        total_moves = 0
        for game in games:
            total_moves += len(game)
        datas = np.zeros([total_moves,16,19,19],  dtype=np.float32)
        labels = np.zeros(total_moves)

        game_start = 0
        for _, game in tqdm(enumerate(games),total=len(games), leave=False):
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


def sort_alternate(array):
    result = np.empty(len(array), dtype=array.dtype)
    result[::2] = np.sort(array[::2])
    result[1::2] = np.sort(array[1::2])
    return result

class WordsDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves, train=True, sort=False, shuffle=False):
        if shuffle:
            games = shuffle_pos(games)
        gamesall = []
        for game in tqdm(games, total = len(games), leave=False):
            result = stepbystep(game)
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
            while(last < num_moves and gamesall[i][last]):
                last += 1
            last -= 1
            y[i] = gamesall[i][last]-1
            gamesall[i][last] = 362
            if sort:
                gamesall[i][:last] = sort_alternate(gamesall[i][:last])
        print("data finish")
        
        gamesall = np.insert(gamesall, 0, 363, axis=1)

        self.x = torch.tensor(gamesall).long()
        self.y = (torch.tensor(y)).long()
        self.mask = (self.x != 0).detach().long()
        self.n_samples = len(self.y)
        
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

def get_datasets(data_config, split_rate=0.1, be_top_left=False, train=True):    
    df = pd.read_csv(data_config["path"], encoding="ISO-8859-1", on_bad_lines='skip')
    df = df.sample(frac=1,replace=False,random_state=17).reset_index(drop=True)\
        .to_numpy()[data_config["offset"]:data_config["offset"]+data_config["data_size"]]
    games = [game for game in df if check(game, data_config["data_source"], data_config["num_moves"])]
    print(f'check_rate:{len(games)/len(df)}')
    print(f'has {len(games)} games')

    if data_config["data_source"] == "foxwq":
        games = np.delete(np.array(games), 0, axis=1)

    games = [[transfer(step) for step in game[:data_config["num_moves"]]] for game in games]
    print("transfer finish")
   
    if be_top_left:
        games = top_left(games)

    split = int(len(games) * split_rate)
    train_dataset = None
    eval_dataset = None
    if data_config["data_type"] == 'Word':
        if train:
            train_dataset = WordsDataset(games[split:],  data_config["num_moves"])
        eval_dataset = WordsDataset(games[:split],  data_config["num_moves"], train=train)
    elif data_config["data_type"] == 'Picture':
        if train:
            train_dataset = PicturesDataset(games[split:], data_config["num_moves"])
        eval_dataset = PicturesDataset(games[:split], data_config["num_moves"])
    elif data_config["data_type"] == "Pretrain":
        games = extend(games)
        train_dataset = BERTPretrainDataset(games, data_config["num_moves"])
        eval_dataset = None
    
    if not train_dataset is None:
        print(f'trainData shape:{train_dataset.x.shape}')
        print(f'trainData memory size:{get_tensor_memory_size(train_dataset.x)}')
    if not eval_dataset is None:
        print(f'evalData shape:{eval_dataset.x.shape}')
        print(f'evalData memory size:{get_tensor_memory_size(eval_dataset.x)}')
    return train_dataset, eval_dataset


if __name__ == "__main__":
    
    
    pass



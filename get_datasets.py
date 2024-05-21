import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
from torch.utils.data import Dataset
from tools import *

class PicturesDataset(Dataset):
    # data loading
    def __init__(self, games):
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
    
        self.x = torch.tensor(datas)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.n_samples = total_moves

        gc.collect()
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class LightPicturesDataset(Dataset):
    # data loading
    def __init__(self, games):
        total_moves = 0
        for game in games:
            total_moves += len(game)
        datas = np.zeros([total_moves,4,19,19],  dtype=np.float32)
        labels = np.zeros(total_moves)
        game_start = 0
        for _, game in tqdm(enumerate(games),total=len(games), leave=False):
            for j, move in enumerate(game):
                labels[game_start] = move
                if j == 0:
                    datas[game_start][2] = np.ones([19,19])
                else:
                    x = int(labels[game_start-1] / 19)
                    y = int(labels[game_start-1] % 19)
                    datas[game_start] = datas[game_start-1]
                    Lchannel_01(datas, game_start, x, y, j)
                    Lchannel_2(datas, game_start, j)
                    Lchannel_3(datas, game_start, x, y, j)
                game_start += 1
    
        self.x = torch.tensor(datas)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.n_samples = total_moves

        gc.collect()
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

class BERTDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves, train=False, sort=False):
        p_dataset = get_board(games)
        gamesall = []
        battle_count = 0
        for game in tqdm(games, total = len(games), leave=False):
            step_games = stepbystep(game, 1)
            if train:
                battle_break = []
                for i in range(1, num_moves):
                    if i > 60 and distance(game[i], game[i-1]) > 5 :
                        battle_break.append(i)
                shuffled_games, count = shuffle_battle(step_games, battle_break)
                battle_count += count
                gamesall.extend(shuffled_games)
            gamesall.extend(step_games)
           
        gamesall = np.array(gamesall)
        print("steps finish")
        print(battle_count)
        total_steps = gamesall.shape[0]
        y = [0]*(total_steps)
        for i in tqdm(range(total_steps), total=total_steps, leave=False):
            last = 0
            while last < num_moves and gamesall[i][last]:
                last += 1
            last -= 1
            y[i] = gamesall[i][last]-1
            gamesall[i][last] = 362
            if sort:
                gamesall[i][:last] = sort_alternate(gamesall[i][:last])
        print("data finish")

        token_types = np.zeros((total_steps, num_moves))
        for i, (game, board) in tqdm(enumerate(zip(gamesall, p_dataset)), total=total_steps, leave=False):
            for j, move in enumerate(game):
                if move == 362:
                    break
                move -= 1
                token_types[i][j] = board[move//19][move%19]

        self.x = torch.tensor(gamesall, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mask = (self.x != 0).detach().long()
        self.token_types = torch.tensor(token_types, dtype=torch.long)
        self.n_samples = len(self.y)
        del p_dataset
        gc.collect()

    def __getitem__(self, index):  
        return self.x[index], self.mask[index], self.token_types[index], self.y[index]

    def __len__(self):
        return self.n_samples

class CombDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves):
        picdata = LightPicturesDataset(games)
        self.xp = picdata.x
        gc.collect()
        bertdata = BERTDataset(games, num_moves)
        self.xw = bertdata.x
        self.mask = bertdata.mask
        self.tokentype = bertdata.token_types
        self.y = bertdata.y
        self.n_samples = bertdata.n_samples
        print(torch.equal(bertdata.y, picdata.y))
        gc.collect()
    def __getitem__(self, index):  
        return self.xw[index], self.mask[index], self.tokentype[index], self.xp[index], self.y[index]

    def __len__(self):
        return self.n_samples

def get_datasets(data_config, split_rate=0.1, train=True):    
    df = pd.read_csv(data_config["path"], encoding="ISO-8859-1", on_bad_lines='skip')
    df = df.sample(frac=1,replace=False,random_state=8596).reset_index(drop=True)\
        .to_numpy()[data_config["offset"]:data_config["offset"]+data_config["data_size"]]
    games = [game for game in df if check(game, data_config["data_source"], data_config["num_moves"])]
    print(f'check_rate:{len(games)/len(df)}')
    print(f'has {len(games)} games')

    if data_config["data_source"] == "foxwq":
        games = np.delete(np.array(games), 0, axis=1)

    games = [[transfer(step) for step in game[:data_config["num_moves"]]] for game in games]
    print("transfer finish")
    split = int(len(games) * split_rate)
    train_dataset = None
    eval_dataset = None

    if data_config["data_type"] == 'Word':
        if train:
            if data_config["extend"]:
                train_dataset = BERTDataset(extend(games[split:]),  data_config["num_moves"], train=train)
            else:
                train_dataset = BERTDataset(games[split:],  data_config["num_moves"], train=train)
        eval_dataset = BERTDataset(games[:split],  data_config["num_moves"])
    elif data_config["data_type"] == 'Picture':
        if train:
            if data_config["extend"]:
                train_dataset = PicturesDataset(extend(games[split:]))
            else:
                train_dataset = PicturesDataset(games[split:])
        eval_dataset = PicturesDataset(games[:split])
    elif data_config["data_type"] == 'LPicture':
        if train:
            if data_config["extend"]:
                train_dataset = LightPicturesDataset(extend(games[split:]))
            else:
                train_dataset = LightPicturesDataset(games[split:])
        eval_dataset = LightPicturesDataset(games[:split])
    elif data_config["data_type"] == "Combine":
        if train:
            train_dataset = CombDataset(games[split:], data_config["num_moves"])
        eval_dataset = CombDataset(games[:split],  data_config["num_moves"])
    
    if data_config["data_type"] == "Combine":
        if not train_dataset is None:
            print(f'trainDatap shape:{train_dataset.xp.shape}')
            print(f'trainDatap memory size:{get_tensor_memory_size(train_dataset.xp)}')
            print(f'trainDataw shape:{train_dataset.xw.shape}')
            print(f'trainDataw memory size:{get_tensor_memory_size(train_dataset.xw)}')
        if not eval_dataset is None:
            print(f'evalDatap shape:{eval_dataset.xp.shape}')
            print(f'evalDatap memory size:{get_tensor_memory_size(eval_dataset.xp)}')
            print(f'evalDataw shape:{eval_dataset.xw.shape}')
            print(f'evalDataaw memory size:{get_tensor_memory_size(eval_dataset.xw)}')
    else:
        if not train_dataset is None:
            print(f'trainData shape:{train_dataset.x.shape}')
            print(f'trainData memory size:{get_tensor_memory_size(train_dataset.x)}')
        if not eval_dataset is None:
            print(f'evalData shape:{eval_dataset.x.shape}')
            print(f'evalData memory size:{get_tensor_memory_size(eval_dataset.x)}')
    gc.collect()
    return train_dataset, eval_dataset


if __name__ == "__main__":
    
    
    pass



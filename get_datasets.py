import numpy as np
import pandas as pd
import copy
import torch
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
        self.y = torch.tensor(labels).long()
        self.n_samples = total_moves
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
class BERTDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves, sort=False, shuffle=False):
        if shuffle:
            games = shuffle_pos(games)
        gamesall = []
        for game in tqdm(games, total = len(games), leave=False):
            result = stepbystep(game)
            gamesall.extend(result)
        gamesall = np.array(gamesall)
        print("steps finish")
        
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
                print("sorted")
                gamesall[i][:last] = sort_alternate(gamesall[i][:last])
        print("data finish")
        gamesall = np.insert(gamesall, 0, 363, axis=1)
        self.x = torch.tensor(gamesall).long()
        self.y = torch.tensor(y).long()
        self.mask = (self.x != 0).detach().long()
        self.n_samples = len(self.y)
        
    def __getitem__(self, index):  
        return self.x[index], self.mask[index], self.y[index]

    def __len__(self):
        return self.n_samples

class CombDataset(Dataset):
    # data loading
    def __init__(self, games, num_moves):
        bertdata = BERTDataset(games, num_moves)
        picdata = PicturesDataset(games)
        print(bertdata.n_samples == picdata.n_samples)
        self.xw = bertdata.x
        self.mask = bertdata.mask
        self.xp = picdata.x
        self.y = bertdata.y
        self.n_samples = bertdata.n_samples

    def __getitem__(self, index):  
        return self.xw[index], self.mask[index], self.xp[index], self.y[index]

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
            games_0 = np.insert(games_0, half, 362, axis=1)
            games_1 = np.insert(games_1, half, 362, axis=1)
            games_0 = np.insert(games_0, length+1, 362, axis=1)
            games_1 = np.insert(games_1, length+1, 362, axis=1)
            a = 0
            b = half+1
            for g, game in enumerate(games_1):
                while(game[a] != 362 and game[b] != 362):
                    games_1[g][a], games_1[g][b] = game[b], game[a]
                    a += 1
                    b += 1
            games = np.concatenate((torch.tensor(games_1), torch.tensor(games_0)), axis=0)
            next_sentence_labels = np.concatenate((torch.ones([games_1.shape[0]]), torch.zeros([games_0.shape[0]])), axis=0)

            # 15% mask data
            labels = copy.deepcopy(games)
            mask = (torch.rand(games.shape) < 0.15) * (games != 0) * (games != 362)
            for i in range(games.shape[0]):
                games[i, torch.flatten(mask[i].nonzero()).tolist()] = 363

            token_type = np.concatenate((torch.zeros([games.shape[0], half+1]), torch.ones([games.shape[0], num_moves+1-half])), axis=1)
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
        total = gamesall[0].shape[0]
        gamesall = np.array(gamesall).reshape(total*(num_moves-1), num_moves+2)
        labels = np.array(labels).reshape(total*(num_moves-1), num_moves+2)
        token_types = np.array(token_types).reshape(total*(num_moves-1), num_moves+2)
        next_sentence_labels = np.array(next_sentence_labels).reshape(total*(num_moves-1))

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
                train_dataset = BERTDataset(extend(games[split:]),  data_config["num_moves"])
            else:
                train_dataset = BERTDataset(games[split:],  data_config["num_moves"])
        eval_dataset = BERTDataset(games[:split],  data_config["num_moves"])
    elif data_config["data_type"] == 'Picture':
        if train:
            if data_config["extend"]:
                train_dataset = PicturesDataset(extend(games[split:]))
            else:
                train_dataset = PicturesDataset(games[split:])
        eval_dataset = PicturesDataset(games[:split])
    elif data_config["data_type"] == "Pretrain":
        games = extend(games)
        train_dataset = BERTPretrainDataset(games, data_config["num_moves"])
        eval_dataset = None
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
    return train_dataset, eval_dataset


if __name__ == "__main__":
    
    
    pass



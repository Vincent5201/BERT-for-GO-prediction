import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from use import next_moves, prediction
from get_datasets import get_datasets
from get_models import get_model
from tools import *

def score_legal(data_type, num_moves, model, device):
    first_steps = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp"]
    moves_score = 0
    score = 0
    full_score = len(first_steps)
    records = []
    for step in tqdm(first_steps, total = len(first_steps), leave = False):
        games = [[step]]
        datas = np.zeros([1,16,19,19],  dtype=np.float32)
        step = transfer(step)
        x = int(step / 19)
        y = int(step % 19)
        channel_01(datas, 0, x, y, len(games[0]))
        channel_2(datas, 0)
        while(len(games[0]) < num_moves):
            next_move = next_moves(data_type, num_moves, model, games, 1, device)[0]
            next_move_ch = transfer_back(next_move)
            x = int(next_move / 19)
            y = int(next_move % 19)
            if datas[0][2][x][y]:
                games[0].append(next_move_ch)
                channel_01(datas, 0, x, y, len(games[0]))
                channel_2(datas, 0)
            else:
                moves_score += len(games[0])
                break
        if len(games[0]) == num_moves:
            score += 1
            moves_score += num_moves
        records.append(games[0])
    return score, moves_score, full_score, records

def score_acc(data_config, model, split, device):
    _, testData = get_datasets(data_config, train=False)
    test_loader = DataLoader(testData, batch_size=64, shuffle=False)
    predl, true = prediction(data_config["data_type"], model, device, test_loader)
    acc5 = myaccn_split(predl, true, 5, split, data_config["num_moves"])
    acc1 = myaccn_split(predl, true, 1, split, data_config["num_moves"])
    return acc5, acc1

def score_self(data_config, model, score_type, device):
    
    if score_type == "score":
        score, moves_score, full_score, records = score_legal(
            data_config["data_type"], data_config["num_moves"], model, device)
        print(records)
        print(f'score:{score}/{full_score}')
        print(f'moves_score:{moves_score/full_score}/{data_config["num_moves"]}')

    elif score_type == "score_acc":
        #use test data
        split = 1
        acc5, acc1 = score_acc(data_config, model, split, device)
        print(acc5)
        print(acc1)

 
if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 3500
    data_config["offset"] = 0
    data_config["data_type"] = "Picture"
    data_config["data_source"] = "pros"
    data_config["num_moves"] = 240

    model_config = {}
    model_config["model_name"] = "ViT"
    model_config["model_size"] = "mid"

    score_type = "score_acc"
    device = "cuda:1"
   
    state = torch.load(f'models_240/ViT1_10000.pt')
    model = get_model(model_config).to(device)
    model.load_state_dict(state)

    score_self(data_config, model, score_type, device)
    
   
    
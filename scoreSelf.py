import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from use import prediction
from myModels import get_model
from myDatasets import get_datasets
from dataAnalyze import plot_board

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

def correct_pos(pred, true):
    correct = [0]*361
    total = [0]*361
    for i, p in tqdm(enumerate(pred), total=len(pred), leave=False):
        sorted_indices = (-p).argsort()
        total[sorted_indices[0]] += 1
        if true[i] == sorted_indices[0]:
            correct[sorted_indices[0]] += 1
    for i in range(361):
        if(total[i]):
            correct[i] /= total[i]
    return correct

def detect_error(data_config, model, device):
    _, testData = get_datasets(data_config, train=False)
    test_loader = DataLoader(testData, batch_size=64, shuffle=False)
    predl, true = prediction(data_config["data_type"], model, device, test_loader)
    datas = testData.x.cpu().numpy()
    total = len(predl)
    error = 0
    correct = 0
    moves = 0
    count_moves = 0
    for i, p in tqdm(enumerate(predl), total=len(predl), leave=False):
        sorted_indices = (-p).argsort()
        x = int(sorted_indices[0]/19)
        y = int(sorted_indices[0]%19)
        if datas[i][2][x][y] == 0:
            error += 1
            moves += i%data_config["num_moves"]
            count_moves += 1
        if sorted_indices[0] == true[i]:
            correct += 1
    return error/total, moves/count_moves, correct/total

def score_acc(data_config, model, split, device):
    batch_size = 64
    _, testData = get_datasets(data_config, train=False)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)
    predl, true = prediction(data_config["data_type"], model, device, test_loader)
    acc10 = myaccn_split(predl, true, 10, split, data_config["num_moves"])
    acc5 = myaccn_split(predl, true, 5, split, data_config["num_moves"])
    acc1 = myaccn_split(predl, true, 1, split, data_config["num_moves"])
    return acc10, acc5, acc1

def correct_position(data_config, model, device):
    batch_size = 64
    _, testData = get_datasets(data_config, train=False)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)
    predl, true = prediction(data_config["data_type"], model, device, test_loader)
    pos = correct_pos(predl, true)
    return pos, myaccn_split(predl, true, 1, 1, data_config["num_moves"])

def score_self(data_config, model, score_type, device):
    
    if score_type == "detect_error":
        rate, acc = detect_error(data_config, model, device)
        print(f'error:{rate}')
        print(f'acc:{acc}')
    elif score_type == "score_acc":
        #use test data
        split = 24
        acc10, acc5, acc1 = score_acc(data_config, model, split, device)
        print(acc10)
        print(acc5)
        print(acc1)
    elif score_type == "correct_pos":
        #use eval data
        pos, acc1 = correct_position(data_config, model, device)
        plot_board(pos)
        print(pos)
        print(acc1)

 
if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 35000
    data_config["offset"] = 0
    data_config["data_type"] = "Picture"
    data_config["data_source"] = "pros"
    data_config["num_moves"] = 240

    model_config = {}
    model_config["model_name"] = "ResNet"
    model_config["model_size"] = "mid"

    score_type = "detect_error"
    device = "cuda:0"
   
    state = torch.load(f'models_{data_config["num_moves"]}/ResNet1_10000.pt')
    model = get_model(model_config).to(device)
    model.load_state_dict(state)
    
    score_self(data_config, model, score_type, device)
    
   
    
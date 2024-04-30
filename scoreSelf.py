import torch
from torch.utils.data import DataLoader

from use import prediction
from get_datasets import get_datasets
from get_models import get_model
from tools import *

def score_acc(data_config, model, split, device):
    _, testData = get_datasets(data_config, train=False)
    test_loader = DataLoader(testData, batch_size=64, shuffle=False)
    predl, true = prediction(data_config["data_type"], model, device, test_loader)
    acc5 = myaccn_split(predl, true, 5, split, data_config["num_moves"])
    acc1 = myaccn_split(predl, true, 1, split, data_config["num_moves"])
    return acc5, acc1

def score_self(data_config, model, score_type, device):  
    if score_type == "score_acc":
        #use test data
        split = 1
        acc5, acc1 = score_acc(data_config, model, split, device)
        print(acc5)
        print(acc1)

if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 35000
    data_config["offset"] = 0
    data_config["data_type"] = "Combine"
    data_config["data_source"] = "pros"
    data_config["num_moves"] = 240
    data_config["extend"] = False

    model_config = {}
    model_config["model_name"] = "Combine"
    model_config["model_size"] = "mid"

    score_type = "score_acc"
    device = "cuda:0"
   
    state = torch.load(f'models/Combine/B10000_R5000.pt')
    model = get_model(model_config).to(device)
    model.load_state_dict(state)
    model.resnet.to(device)
    model.bert.to(device)

    score_self(data_config, model, score_type, device)
    
   
    
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

from use import prediction
from get_datasets import get_datasets
from get_models import get_model
from tools import myaccn_split

def score_acc(data_config, model, split, device):
   # _, testData = get_datasets(data_config, train=False)
    #test_loader = DataLoader(testData, batch_size=64, shuffle=False)
    #predl, true = prediction(data_config["data_type"], model, device, test_loader)
    predls = np.load('analyze_data/predls4.npy')
    true = np.load('analyze_data/trues4.npy')
    predl = torch.tensor(predls[2])
    acc5 = myaccn_split(predl, true, 5, split, data_config["num_moves"])
    acc1 = myaccn_split(predl, true, 1, split, data_config["num_moves"])
    return acc5, acc1

def scores(data_config, model, device):
    
    _, testData = get_datasets(data_config, train=False)
    test_loader = DataLoader(testData, batch_size=64, shuffle=False)
    predl, true = prediction(data_config["data_type"], model, device, test_loader)
    
    predls = np.load('analyze_data/predls4.npy')
    true = np.load('analyze_data/trues4.npy')
    
    predl = torch.tensor(predls[3])
    
    predl = torch.tensor(predl)
    preds = torch.max(predl,1).indices

    print(f'accuracy_socre: {accuracy_score(true, preds)}')
    print(f'f1_socre: {f1_score(true, preds, average="micro")}')
    print(f'precision_socre: {precision_score(true, preds, average="micro")}')
    print(f'recall_socre: {recall_score(true, preds, average="micro")}')


def score_self(data_config, model, score_type, device):  
    if score_type == "score_acc":
        #use test data
        split = 1
        acc5, acc1 = score_acc(data_config, model, split, device)
        print(acc5)
        print(acc1)
    if score_type == "scores":
        scores(data_config, model, device)



if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_Foxwq_9d.csv'
    data_config["data_size"] = 50000
    data_config["offset"] = 0
    data_config["data_type"] = "Word"
    data_config["data_source"] = "foxwq" 
    data_config["num_moves"] = 240
    data_config["extend"] = False

    model_config = {}
    model_config["model_name"] = "BERT"
    model_config["model_size"] = "mid"

    score_type = "score_acc"
    device = "cuda:1"
   
    state = torch.load(f'models/BERTex/mid_s63_5000.pt')
    model = get_model(model_config).to(device)
    model.load_state_dict(state)
    #model.m1 = model.m1.to(device)
    #model.m2 = model.m2.to(device)

    score_self(data_config, model, score_type, device)
    
   
    
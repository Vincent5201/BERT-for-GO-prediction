import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random

from use import prediction
from get_datasets import get_datasets
from get_models import get_model
from tools import *

def class_correct_moves(predls, trues, n):
    records = [[] for _ in range(2**len(predls))]
    print("start classiy")
    for j, tgt in tqdm(enumerate(trues), total=len(trues), leave=False):
        pos = 0
        for i, predl in enumerate(predls):
            pos *= 2
            sorted_indices = (-(predl[j])).argsort()
            top_k_indices = sorted_indices[:n]  
            if tgt in top_k_indices:
                pos += 1
        records[pos].append(j)
    return records

def prob_vote(sorted_indices, sorted_p):
    vote = {}
    for p, indices in enumerate(sorted_indices):
        for q, indice in enumerate(indices):
            if indice in vote.keys():
                if p:
                    vote[indice] += sorted_p[p][q]
                else:
                    vote[indice] += sorted_p[p][q]*3
            else:
                if p:
                    vote[indice] = sorted_p[p][q]
                else:
                    vote[indice] = sorted_p[p][q]*3
    sorted_vote = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
    choices = list(sorted_vote.keys())
    return choices

def prob_rank_vote(sorted_indices, sorted_p):
    prob_vote = {}
    rank_vote = {}
    for p, indices in enumerate(sorted_indices):
        for q, indice in enumerate(indices):
            if indice in prob_vote.keys():
                if p:
                    prob_vote[indice] += sorted_p[p][q]
                else:
                    prob_vote[indice] += sorted_p[p][q]*3
            else:
                if p:
                    prob_vote[indice] = sorted_p[p][q]
                else:
                    prob_vote[indice] = sorted_p[p][q]*3
            if indice in rank_vote.keys():
                if p:
                    rank_vote[indice] += (1-q/10)
                else:
                    rank_vote[indice] += (0.95-q/10)
            else:
                if p:
                    rank_vote[indice] = (1-q/10)
                else:
                    rank_vote[indice] = (0.95-q/10)
    sorted_prob = dict(sorted(prob_vote.items(), key=lambda item: item[1], reverse=True))
    sorted_rank = dict(sorted(rank_vote.items(), key=lambda item: item[1], reverse=True))
    probs = list(sorted_prob.keys())[:5]
    ranks = list(sorted_rank.keys())[:5]
    comb_vote = {}
    for i, indice in enumerate(probs):
        if indice in comb_vote.keys():
            comb_vote[indice] += (1-i*i/10)
        else:
            comb_vote[indice] = (1-i*i/10)
    for i, indice in enumerate(ranks):
        if indice in comb_vote.keys():
            comb_vote[indice] += (0.95-i*i/10)
        else:
            comb_vote[indice] = (0.95-i*i/10)
    sorted_comb = dict(sorted(comb_vote.items(), key=lambda item: item[1], reverse=True))
    combs = list(sorted_comb.keys())[:5]
    return combs

def get_data_pred(data_config, models, data_types, device):
    batch_size = 64
    predls = []
    if "Word" in data_types:
        data_config["data_type"] = "Word"
        _, testDataW = get_datasets(data_config, train=False)
        test_loaderW = DataLoader(testDataW, batch_size=batch_size, shuffle=False)
        trues = testDataW.y
    if "Picture" in data_types:
        data_config["data_type"] = "Picture"
        _, testDataP = get_datasets(data_config, train=False)
        test_loaderP = DataLoader(testDataP, batch_size=batch_size, shuffle=False)
        trues = testDataP.y
    if "Combine" in data_types:
        data_config["data_type"] = "Combine"
        _, testDataWP = get_datasets(data_config, train=False)
        test_loaderWP = DataLoader(testDataWP, batch_size=batch_size, shuffle=False)
        trues = testDataW.y

    for i, model in enumerate(models):
        if data_types[i] == "Word":
            predl, _ = prediction("Word", model, device, test_loaderW)
        elif data_types[i] == "Combine":
            predl, _ = prediction("Combine", model, device, test_loaderWP)
        elif data_types[i] == "Picture":
            predl, _ = prediction("Picture", model, device, test_loaderP)
        predls.append(predl)
    
    #np.save('analyze_data/predls2.npy', predls)
    #np.save('analyze_data/trues3.npy', trues)
    return testDataP, testDataW, predls, trues.cpu().numpy()

def mix_acc(n, predls, trues, smart=None):
    total = len(trues)
    correct = 0
    for i in tqdm(range(total), total=total, leave=False):
        sorted_indices = []
        sorted_p = []
        for _, predl in enumerate(predls):
            sorted_indices.append((-predl[i]).argsort()[:10]) 
            sorted_p.append(np.sort(predl[i])[::-1][:10])
        
        choices = []
        if smart == "prob_vote":
            choices = prob_vote(sorted_indices, sorted_p)
        elif smart == "prob_rank_vote":
            choices = prob_rank_vote(sorted_indices, sorted_p)
        else:
            choices = [s[0] for s in sorted_indices]
            random.shuffle(choices)
        if trues[i] in choices[:n]:
            correct += 1
    return correct/total

def compare_correct(predls, trues, n):
    record1 = class_correct_moves(predls, trues, n)
    total = len(trues)
    count = [len(record)/total for record in record1]

    return record1, count

def invalid_rate(board, predls, n=1):
    total = len(predls[0])
    invalid = [0]*len(predls)
    for i in tqdm(range(total), total=total, leave=False):
        for j, predl in enumerate(predls):
            chooses = (-predl[i]).argsort()[:n]
            check = True
            for c in chooses:
                if board[i][2][int(c/19)][int(c%19)]:
                    check = False
                    break
            if check:
                invalid[j] += 1
    return [e/total for e in invalid]
            

def score_more(data_config, models, device, score_type):
    testDataP = None
    testDataP, testDataW, predls, trues = get_data_pred(data_config, models, data_types, device)
    #predls = np.load('analyze_data/predls2.npy')
    #trues = np.load('analyze_data/trues.npy')

    if score_type == "compare_correct":
        records, count = compare_correct(predls, trues, 1)
        print(count)
        #print(records)
    elif score_type == "mix_acc":
        acc = mix_acc(1, predls, trues, "prob_vote")
        print(acc)
    elif score_type == "invalid":
        invalid = invalid_rate(testDataP.x, predls, 1)
        print(invalid)
        invalid = invalid_rate(testDataP.x, predls, 5)
        print(invalid)

if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_Foxwq_9d.csv'
    data_config["data_size"] = 50000
    data_config["offset"] = 0
    data_config["data_type"] = "Picture"
    data_config["data_source"] = "foxwq"
    data_config["num_moves"] = 240
    data_config["extend"] = False

    model_config = {}
    model_config["model_name"] = "ST"
    model_config["model_size"] = "mid"

    device = "cuda:0"
    score_type = "compare_correct"

    data_types = ['Picture', 'Word','Picture', 'Word']
    model_names = ["ResNet", "BERT","ResNet", "BERT"] #abc
    states = [f'models/ResNet/mid_s12_1600.pt',
              f'models/BERT/mid_s59_10000.pt',
              f'models/ResNet/mid_s57_5000.pt',
              f'models/BERT/mid_s27_30000.pt']
    models = []
    for i in range(len(model_names)):
        model_config["model_name"] = model_names[i]
        model = get_model(model_config).to(device)
        state = torch.load(states[i])
        model.load_state_dict(state)
        models.append(model)

    #get_data_pred(data_config, models, data_types, device)
    score_more(data_config, models, device, score_type)
   
    
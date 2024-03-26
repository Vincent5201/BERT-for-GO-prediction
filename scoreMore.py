import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy
import math
import random
import yaml

from use import next_moves, prediction
from myModels import get_model
from myDatasets import transfer_back, channel_01, channel_2, channel_1015, get_datasets

def score_legal_more(data_types, num_moves, models):
    first_steps = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp","cc", "cq", "qc","qq"]
    scores = [0]*len(models)
    moves_scores = [0]*len(models)
    errors = [0]*len(models)
    full_score = len(first_steps) * len(models)
    records = []
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            for step in first_steps:
                full_score += 1
                games = [[step]]
                while(len(games[0]) < num_moves):
                    if(len(games[0])%2):
                        next_move = next_moves(data_types[i], num_moves, model1, games, 1)[0]
                        next_move = transfer_back(next_move)
                    else:
                        next_move = next_moves(data_types[j], num_moves, model2, games, 1)[0]
                        next_move = transfer_back(next_move)
                    if next_move in games[0]:
                        if(len(games[0])%2):
                            errors[i] += 1
                        else:
                            errors[j] += 1
                        moves_scores[i] += len(games[0])
                        moves_scores[j] += len(games[0])
                        break
                    else:
                        games[0].append(next_move)

                if len(games[0]) == num_moves:
                    scores[i] += 1
                    scores[j] += 1
                    moves_scores[i] += num_moves
                    moves_scores[j] += num_moves
                records.append(games[0])
    return scores, moves_scores, full_score, errors, records

def class_correct_moves(predls, true, n):
    records = [[] for _ in range(8)]
    print("start classiy")
    for j in tqdm(range(len(true)), total=len(true), leave=False):
        pos = 0
        for i in range(3):
            pos *= 2
            sorted_indices = (-(predls[i][j])).argsort()
            top_k_indices = sorted_indices[:n]  
    
            if true[j] in top_k_indices:
                pos += 1
        records[pos].append(j)
    return records

def check_liberty(inp, trues, tgt, num_move):
    x = int((trues[tgt]-1)/19)
    y = int((trues[tgt]-1)%19)
    plane = np.expand_dims(copy.deepcopy(np.array(inp[tgt])), axis=0)
    channel_01(plane, 0, x, y, tgt%num_move+1)
    channel_2(plane, 0)
    return channel_1015(plane, 0, x, y, tgt%num_move+1)

def check_distance(trues, tgt):
    if tgt%80:
        x = int((trues[tgt]-1)/19)
        y = int((trues[tgt]-1)%19)
        xl = int((trues[tgt-1]-1)/19)
        yl = int((trues[tgt-1]-1)%19)
        return math.sqrt(math.pow(x-xl,2)+math.pow(y-yl,2))
    return None
 
def vote_move(sorted_indices, num_move):
    top_choices = [p[0] for p in sorted_indices]
    if len(top_choices) == len(set(top_choices)):
        if i%num_move < 40:
            choices = [top_choices[2]]
        elif i%num_move > 190:
            choices = [top_choices[1]]
        else:
            choices = [top_choices[0]]
    else:
        if len(set(top_choices)) == 1:
            choices = [top_choices[0]]
        else:
            if top_choices[0] == top_choices[1]:
                choices = [top_choices[0]]
            elif top_choices[0] == top_choices[2]:
                choices = [top_choices[0]]
            else:
                choices = [top_choices[1]]
    return choices

def vote_prec(sorted_indices, precs):    
    mat1, mat2, mat3 = precs
    tops = [mat1[sorted_indices[0][0]], mat2[sorted_indices[1][0]], mat3[sorted_indices[2][0]]]
    top_choices = [p[0] for p in sorted_indices]

    if len(top_choices) == len(set(top_choices)):
        if tops[0] > tops[1]:
            if tops[0] > tops[2]:
                choices = [top_choices[0]]
            else:
                choices = [top_choices[2]]
        else:
            if tops[1] > tops[2]:
                choices = [top_choices[1]]
            else:
                choices = [top_choices[2]]
    else:
        if len(set(top_choices)) == 1:
            choices = [top_choices[0]]
        else:
            if top_choices[0] == top_choices[1]:
                choices = [top_choices[0]]
            elif top_choices[0] == top_choices[2]:
                choices = [top_choices[0]]
            else:
                choices = [top_choices[1]]
    return choices

def vote_model(sorted_indices, choose):
    top_choices = [p[0] for p in sorted_indices]
    if len(top_choices) == len(set(top_choices)):
        choices = [top_choices[choose]]
    else:
        if len(set(top_choices)) == 1:
            choices = [top_choices[0]]
        else:
            if top_choices[0] == top_choices[1]:
                choices = [top_choices[0]]
            elif top_choices[0] == top_choices[2]:
                choices = [top_choices[0]]
            else:
                choices = [top_choices[1]]
    return choices

def prob_vote(sorted_indices, sorted_p):
    vote = {}
    for p, indices in enumerate(sorted_indices):
        for q, indice in enumerate(indices):
            if indice in vote.keys():
                vote[indice] += sorted_p[p][q]
            else:
                vote[indice] = sorted_p[p][q]
    sorted_vote = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
    choices = list(sorted_vote.keys())
    return choices

def vote(sorted_indices):
    vote = {}
    for p, indices in enumerate(sorted_indices):
        for q, indice in enumerate(indices):
            if indice in vote.keys():
                vote[indice] += (1-q*q/10)
            else:
                vote[indice] = (1-q*q/10)
    sorted_vote = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
    choices = list(sorted_vote.keys())
    return choices

def get_data_pred(data_config, models, data_types, device):
    batch_size = 64
    predls = []
    data_config["data_type"] = "Word"
    _, testDataW = get_datasets(data_config, train=False)
    test_loaderW = DataLoader(testDataW, batch_size=batch_size, shuffle=False)
    trues = testDataW.y

    data_config["data_type"] = "Picture"
    _, testDataP = get_datasets(data_config, train=False)
    test_loaderP = DataLoader(testDataP, batch_size=batch_size, shuffle=False)

    for i, model in enumerate(models):
        if data_types[i] == "Word":
            predl, _ = prediction(data_types[i], model, device, test_loaderW)
        else:
            predl, _ = prediction(data_types[i], model, device, test_loaderP)
        predls.append(predl)
    np.save('analyzation_data/prediction.npy', np.array(predls))
    np.save('analyzation_data/trues.npy', trues.cpu().numpy())

    return testDataP, testDataW, predls, trues.cpu().numpy()

def mix_acc(n, data_config, device, models, data_types, smart=None):
    
    #_, _, predls, trues = get_data_pred(data_config, models, data_types, device)
    predls = np.load('analyzation_data/prediction_35000_240.npy')
    trues = np.load('analyzation_data/trues_35000_240.npy')
    
    print("pred end")
    with open('analyzation.yaml', 'r') as file:
        args = yaml.safe_load(file)
    precs = [args["pos_recall"][f'model_{data_config["num_moves"]}']["ResNet"],
             args["pos_recall"][f'model_{data_config["num_moves"]}']["ViT"],
             args["pos_recall"][f'model_{data_config["num_moves"]}']["ST"]]
    
    total = len(trues)
    correct = 0
    for i in tqdm(range(total), total=total, leave=False):
        sorted_indices = []
        sorted_p = []
        for _, predl in enumerate(predls):
            sorted_indices.append((-predl[i]).argsort()[:5]) 
            sorted_p.append(np.sort(predl[i])[::-1][:5])
        
        choices = []
        if smart == "prob_vote":
            choices = prob_vote(sorted_indices, sorted_p)
        elif smart == "vote+ResNet":
            choices = vote_model(sorted_indices, 0)
        elif smart == "vote+ViT":
            choices = vote_model(sorted_indices, 1)
        elif smart == "vote+prec":
            choices = vote_prec(sorted_indices, precs)
        elif smart == "vote":
            choices = vote(sorted_indices)
        else:
            choices = [s[0] for s in sorted_indices]
            random.shuffle(choices)

        if trues[i] in choices[:n]:
            correct += 1
    return correct/total

def compare_correct(data_config, device, models, data_types):

    testDataP, testDataW, predls, trues = get_data_pred(data_config, models, data_types, device)
    
    record1 = class_correct_moves(predls, trues, 1)
    total = len(trues)

    count = [len(record)/total for record in record1]

    move_nums = [[move%data_config["num_moves"] for move in record] for record in record1]
    avg_move_num = [sum(move_num)/len(move_num) if len(move_num) else 0 for move_num in move_nums ]
    
    avg_liberty = [0]*len(record1)
    for i, record in tqdm(enumerate(record1), total=len(record1)):
        for move in record:
            avg_liberty[i] += check_liberty(testDataP.x, testDataP.y, move, data_config["num_moves"])
    avg_liberty = [liberty/len(record1[i]) if len(record1[i]) else 0 for i, liberty in enumerate(avg_liberty)]
    avg_distance = [[check_distance(trues, move) for move in record] for record in record1]
    avg_distance = [[distance for distance in distances if not distance is None] for distances in avg_distance]
    avg_distance = [sum(distance)/len(distance) if len(distance) else 0 for distance in avg_distance]
    
    return record1, count, avg_move_num, avg_liberty, avg_distance


def score_more(data_config, models, device, score_type):
        
    if score_type == "legal":
        score, moves_score, full_score, errors = score_legal_more(
            data_types, data_config["num_moves"], models)
        print(f'score:{score}/{full_score}')
        print(f'moves_score:{moves_score}/{full_score*data_config["num_moves"]}')
        print(f'error:{errors}')

    elif score_type == "compare_correct":
        records, count, avg_move_num, avg_liberty, avg_distance = compare_correct(
            data_config, device, models, data_types)
        print(count)
        print(avg_move_num)
        print(avg_liberty)
        print(avg_distance)
        #print(records)

    elif score_type == "mix_acc":
        acc = mix_acc(1, data_config, device, models, data_types, "vote+ViT")
        print(acc)

 
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

    device = "cuda:1"
    score_type = "mix_acc"

    data_types = ['Picture', 'Picture', 'Picture']
    model_names = ["ResNet", "ViT", "ST"] #abc
    states = [f'models_{data_config["num_moves"]}/ResNet1_10000.pt',
              f'models_{data_config["num_moves"]}/ViT1_10000.pt',
              f'models_{data_config["num_moves"]}/ST1_10000.pt']
    models = []
    for i in range(len(model_names)):
        model_config["model_name"] = model_names[i]
        model = get_model(model_config).to(device)
        state = torch.load(states[i])
        model.load_state_dict(state)
        models.append(model)

    score_more(data_config, models, device, score_type)
   
    
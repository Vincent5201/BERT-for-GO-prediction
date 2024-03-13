import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import copy
import math
import random
from use import next_moves
from myModels import get_model
from myDatasets import transfer_back, channel_01, channel_2, transfer, channel_1015, get_datasets
from dataAnalyze import plot_board




def score_legal_self(data_type, num_moves, model):
    first_steps = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp"]
    moves_score = 0
    score = 0
    full_score = len(first_steps)
    records = []
    for step in tqdm(first_steps, total = len(first_steps), leave = False):
        games = [[step]]
        datas = np.zeros([1,16,19,19],  dtype=np.float32)
        f_step = transfer(step)
        x = int(f_step / 19)
        y = int(f_step % 19)
        channel_01(datas, 0, x, y, len(games[0]))
        channel_2(datas, 0)
        while(len(games[0]) < num_moves):
            next_move = next_moves(data_type, num_moves, model, games, 1)[0]
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



def score_feature_self(data_type, num_moves, model, bounds):
    first_steps = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp"]
    nears = [0]*len(bounds)
    total = 0
    atari = 0
    liberty = 0
    records = []
    for step in tqdm(first_steps, total = len(first_steps), leave = False):
        games = [[step]]
        datas = np.zeros([1,16,19,19],  dtype=np.float32)
        f_step = transfer(step)
        x = int(f_step / 19)
        y = int(f_step % 19)
        lastx = x
        lasty = y
        channel_01(datas, 0, x, y, len(games[0]))
        channel_2(datas, 0)
        liberty += channel_1015(datas, 0, x, y, len(games[0]))
        while(len(games[0]) < num_moves):
            next_move = next_moves(data_type, num_moves, model, games, 1)[0]
            next_move_ch = transfer_back(next_move)
            x = int(next_move / 19)
            y = int(next_move % 19)
            if datas[0][2][x][y]:
                games[0].append(next_move_ch)
                channel_01(datas, 0, x, y, len(games[0]))
                channel_2(datas, 0)
                liberty += channel_1015(datas, 0, x, y, len(games[0]))
                total += 1
                # distance
                for i, bound in enumerate(bounds):
                    if (pow(lastx-x, 2) + pow(lasty-y, 2)) < bound*bound:
                        nears[i] += 1
                lastx = x
                lasty = y
                # atari
                p = 1
                if len(games[0]) % 2:
                    p = 0
                if x > 0 and datas[0][p][x-1][y] and datas[0][10][x-1][y]:
                    atari += 1
                if y > 0 and datas[0][p][x][y-1] and datas[0][10][x][y-1]:
                    atari += 1
                if x < 18 and datas[0][p][x+1][y] and datas[0][10][x+1][y]:
                    atari += 1
                if y < 18 and datas[0][p][x][y+1] and datas[0][10][x][y+1]:
                    atari += 1
            else:
                games[0].append(next_move_ch)
                break
        records.append(games[0])
    return [near/total for near in nears], atari/total, liberty/total



def prediction(data_type, model, device, test_loader):
    model.eval()
    preds = []
    predl = []
    true = []
    with torch.no_grad():
        for datas in tqdm(test_loader, leave=False):
            if data_type == "Word":
                x, m, y = datas
                x = x.to(device)
                m = m.to(device)
                y = y.to(device)
                pred = model(x, m)
            elif data_type == "Picture":
                x, y = datas
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            pred = torch.nn.functional.softmax(pred, dim=1)
            predl += pred
            ans = torch.max(pred,1).indices
            preds += ans
            true += y
    predl = torch.stack(predl).cpu().numpy()
    true = torch.stack(true)
    true = torch.tensor(true).cpu().numpy()
    preds = torch.tensor(preds).cpu().numpy()
    return predl, true

def myaccn_split(pred, true, n, split, num_move):
    total = len(true)
    correct = [0]*split
    for i, p in tqdm(enumerate(pred), total=len(pred), leave=False):
        sorted_indices = (-p).argsort()
        top_k_indices = sorted_indices[:n]  
        if true[i] in top_k_indices:
            correct[int((i%num_move)/int(num_move/split))] += 1
    for i in range(split):
        correct[i] /= (total/split)
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

def score_acc(num_moves, data_type, model, split, data_size):
    
    batch_size = 64
    path = 'datas/data_240119.csv'
    data_source = "pros" 
    split_rate = 0.1
    _, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate
                                ,be_top_left=False, train=False)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)
    predl, true = prediction(data_type, model, device, test_loader)
    acc10 = myaccn_split(predl,true,10,split, num_moves)
    acc5 = myaccn_split(predl,true,5,split, num_moves)
    acc1 = myaccn_split(predl,true,1,split, num_moves)
    return acc10, acc5, acc1

def correct_position(num_moves, data_type, model, data_size):
    
    batch_size = 64
    path = 'datas/data_240119.csv'
    data_source = "pros" 
    split_rate = 0.1
    _, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate
                                ,be_top_left=False, train=False)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)
    predl, true = prediction(data_type, model, device, test_loader)
    pos = correct_pos(predl, true)
    return pos, myaccn_split(predl,true,1,1,num_moves)

def score_self(num_moves, model_size, device, score_type):
    data_type = "Picture"
    model_name = "ResNet"
    state = torch.load('models_160/ResNet1_15000.pt')
    model = get_model(model_name, model_size).to(device)
    model.load_state_dict(state)

    if score_type == "score":
        score, moves_score, full_score, records = score_legal_self(data_type, num_moves, model)
        print(records)
        print(f'score:{score}/{full_score}')
        print(f'moves_score:{moves_score/full_score}/{num_moves}')
    elif score_type == "feature":
        bounds = [1.5, 2.9, 4.3, 5.7, 7.1, 8.5]
        near, atari, liberty = score_feature_self(data_type, num_moves, model, bounds)
        print(f'near:{near}')
        print(f'atari:{atari}')
        print(f'liberty:{liberty}')
    elif score_type == "score_acc":
        split = 16
        data_size = 35000
        acc10, acc5, acc1 = score_acc(num_moves, data_type, model, split, data_size)
        print(acc10)
        print(acc5)
        print(acc1)
    elif score_type == "correct_pos":
        data_size = 35000
        pos, acc1 = correct_position(num_moves, data_type, model, data_size)
        plot_board(pos)
        print(pos)
        print(acc1)

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
    plane = copy.deepcopy(np.array(inp[tgt]))
    plane = np.expand_dims(plane, axis=0)
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

def mix_acc_split(predls, true, n, split, num_move, smart=True):
    total = len(true)
    correct = [0]*split
    
    for i in tqdm(range(total), total=total, leave=False):
        sorted_indices = []
        sorted_p = []
        for j, predl in enumerate(predls):
            sorted_indices.append((-predl[i]).argsort()[:min(3*n,10)]) 
            sorted_p.append(np.sort(predl[i])[::-1][:min(3*n,10)])
        sorted_indices = np.array(sorted_indices)
        sorted_p = np.array(sorted_p)
        sorted_indices = sorted_indices.reshape(sorted_indices.shape[0]*sorted_indices.shape[1])
        sorted_p = sorted_p.reshape(sorted_p.shape[0]*sorted_p.shape[1])
        vote = {}
        if smart:
            #acc:49.2 for models_160
            for j, idx in enumerate(sorted_indices):
                for k in range(j+1, len(sorted_indices)):
                    if idx == sorted_indices[k]:
                        sorted_p[j] += sorted_p[k]
            sorted_idx = np.argsort(sorted_p)[::-1]
            choices = sorted_indices[sorted_idx][:n]
            """
            #acc:48.2 for models_160
            for idx in sorted_indices:
                if idx in vote.keys():
                    vote[idx] += 1
                else:
                    vote[idx] = 1
            sorted_vote = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
            max_vote = 0
            choices = []
            for item in sorted_vote.items():
                if max_vote:
                    if item[1] == max_vote:
                        choices.append(item[0])
                    else:
                        break
                else:
                    choices.append(item[0])
                    max_vote = item[1]
           # random.shuffle(choices)
            choices = choices[:n]
        else:
            sorted_idx = np.argsort(sorted_p)[::-1]
            choices = sorted_indices[sorted_idx][:n]
        """
        if true[i] in choices:
            correct[int((i%num_move)/int(num_move/split))] += 1
    for i in range(split):
        correct[i] /= (total/split)
    return correct 

def compare_correct(num_moves, device, models, data_types, data_size):

    batch_size = 64
    path = 'datas/data_240119.csv'
    data_source = "pros" 
    split_rate = 0.1
    be_top_left = False
    predls = []
    _, testDataW = get_datasets(path, "Word", data_source, data_size, num_moves, split_rate
                                , be_top_left, train=False)
    _, testDataP = get_datasets(path, "Picture", data_source, data_size, num_moves, split_rate
                                , be_top_left, train=False)
    for i, model in enumerate(models):
        if data_types[i] == "Word":
            test_loader = DataLoader(testDataW, batch_size=batch_size, shuffle=False)
        else:
            test_loader = DataLoader(testDataP, batch_size=batch_size, shuffle=False)
        predl, true = prediction(data_types[i], model, device, test_loader)
        predls.append(predl)
    record1 = class_correct_moves(predls, true, 1)
    total = len(true)

    acc = mix_acc_split(predls, true, 1, 1, num_moves)
    print(acc)
    count = [len(record)/total for record in record1]

    move_nums = [[move%num_moves for move in record] for record in record1]
    avg_move_num = [sum(move_num)/len(move_num) if len(move_num) else 0 for move_num in move_nums ]
    
    avg_liberty = [0]*len(record1)
    for i, record in tqdm(enumerate(record1), total=len(record1)):
        for move in record:
            avg_liberty[i] += check_liberty(testDataP.x, testDataP.y, move, num_moves)
    avg_liberty = [liberty/len(record1[i]) if len(record1[i]) else 0 for i, liberty in enumerate(avg_liberty)]
    avg_distance = [[check_distance(true, move) for move in record] for record in record1]
    avg_distance = [[distance for distance in distances if not distance is None] for distances in avg_distance]
    avg_distance = [sum(distance)/len(distance) if len(distance) else 0 for distance in avg_distance]
    
    return record1, acc, count, avg_move_num, avg_liberty, avg_distance

    

def score_more(num_moves, model_size, device, score_type):
    
    data_types = ['Picture', 'Picture', 'Picture']
    model_names = ["ResNet", "ResNet", "ResNet"] #abc
    states = ['models_240/ResNet1_1600.pt',
              'models_240/ResNet11_1600.pt',
              'models_240/ResNet111_1600.pt']
    
    models = []
    for i in range(len(model_names)):
        model = get_model(model_names[i], model_size).to(device)
        state = torch.load(states[i])
        model.load_state_dict(state)
        models.append(model)
        
    if score_type == "legal":
        score, moves_score, full_score, errors = score_legal_more(data_types, num_moves, models)
        print(f'score:{score}/{full_score}')
        print(f'moves_score:{moves_score}/{full_score*num_moves}')
        print(f'error:{errors}')
    elif score_type == "correct_compare":
        data_size = 35000
        records, acc, count, avg_move_num, avg_liberty, avg_distance = compare_correct(
            num_moves, device, models, data_types, data_size)
        print(acc)
        print(count)
        print(avg_move_num)
        print(avg_liberty)
        print(avg_distance)
        #print(records)

 
if __name__ == "__main__":
    num_moves = 160
    model_size = "mid"
    device = "cuda:1"
    score_type = "correct_pos"

    score_self(num_moves, model_size, device, score_type)
    #score_more(num_moves, model_size, device, score_type)
   
    
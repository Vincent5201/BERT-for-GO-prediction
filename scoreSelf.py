import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from use import next_moves, prediction
from myModels import get_model
from myDatasets import transfer_back, channel_01, channel_2, transfer, channel_1015, get_datasets
from dataAnalyze import plot_board

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



def score_feature(data_type, num_moves, model, bounds):
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

def score_acc(num_moves, data_type, model, split, data_size, device):
    
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

def correct_position(num_moves, data_type, model, data_size, device):
    
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

def score_self(num_moves, model, score_type, device):
    
    if score_type == "score":
        score, moves_score, full_score, records = score_legal(data_type, num_moves, model, device)
        #print(records)
        print(f'score:{score}/{full_score}')
        print(f'moves_score:{moves_score/full_score}/{num_moves}')
    elif score_type == "feature":
        bounds = [1.5, 2.9, 4.3, 5.7, 7.1, 8.5]
        near, atari, liberty = score_feature(data_type, num_moves, model, bounds)
        print(f'near:{near}')
        print(f'atari:{atari}')
        print(f'liberty:{liberty}')
    elif score_type == "score_acc":
        split = 16
        data_size = 35000
        acc10, acc5, acc1 = score_acc(num_moves, data_type, model, split, data_size, device)
        print(acc10)
        print(acc5)
        print(acc1)
    elif score_type == "correct_pos":
        # use eval data
        data_size = 30000
        pos, acc1 = correct_position(num_moves, data_type, model, data_size, device)
        plot_board(pos)
        print(pos)
        print(acc1)

 
if __name__ == "__main__":
    num_moves = 160
    model_size = "mid"
    device = "cuda:0"
    data_type = "Word"
    model_name = "BERTp"
    state = torch.load(f'models_{num_moves}/BERT11_15000_15000.pt')
    model = get_model(model_name, model_size).to(device)
    model.load_state_dict(state)
    score_type = "score"
    score_self(num_moves, model, score_type, device)
    
   
    
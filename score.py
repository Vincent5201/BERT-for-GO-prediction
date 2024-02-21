import torch
import numpy as np
from tqdm import tqdm
from use import next_moves
from myModels import get_model
from myDatasets import transfer_back, channel_01, channel_2, transfer, channel_1015


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


if __name__ == "__main__":
    """
    data_type = 'Picture'
    num_moves = 80
    model = get_model("ST", "mid").to("cuda:1")
    state = torch.load('/home/F74106165/Transformer_Go/models/ST1.pt')
    model.load_state_dict(state)
    bounds = [1.5, 2.9, 4.3, 5.7, 7.1, 8.5]
    near, atari, liberty = score_feature_self(data_type, num_moves, model, bounds)
    print(f'near:{near}')
    print(f'atari:{atari}')
    print(f'liberty:{liberty}')
    """
    # score self
    data_type = 'Picture'
    num_moves = 80
    model = get_model("ResNet", "mid").to("cuda:1")
    state = torch.load('/home/F74106165/Transformer_Go/models_80/ResNet1_1600.pt')
    model.load_state_dict(state)
    score, moves_score, full_score, records = score_legal_self(data_type, num_moves, model)
    print(f'score:{score}/{full_score}')
    print(f'moves_score:{moves_score/full_score}/{num_moves}')
    #print(records[1])
    
    """
    #score more
    data_types = ['Word', 'Picture', 'Picture', 'Picture']
    num_moves = 80
    model1 = get_model("BERT", 0).to("cuda:1")
    state1 = torch.load('/home/F74106165/go_data/BERT/models/BERT0.pt')
    model1.load_state_dict(state)
    model2 = get_model("ResNet", 0).to("cuda:1")
    state2 = torch.load('/home/F74106165/go_data/BERT/models/BERT0.pt')
    model2.load_state_dict(state)
    model3 = get_model("Vit", 0).to("cuda:1")
    state3 = torch.load('/home/F74106165/go_data/BERT/models/BERT0.pt')
    model3.load_state_dict(state)
    model4 = get_model("ResNetxViT", 0).to("cuda:1")
    state4 = torch.load('/home/F74106165/go_data/BERT/models/BERT0.pt')
    model4.load_state_dict(state)

    models = [model1, model2, model3, model4]
    score, moves_score, full_score, errors = score_legal_more(data_types, num_moves, models)
    print(f'score:{score}/{full_score}')
    print(f'moves_score:{moves_score}/{full_score*num_moves}')
    print(f'error:{errors}')
    """
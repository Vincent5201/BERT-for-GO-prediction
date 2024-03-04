import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from use import next_moves
from myModels import get_model
from myDatasets import transfer_back, channel_01, channel_2, transfer, channel_1015, get_datasets

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

def compare_acc(predls, true, n):
    
    total = len(true)
    record = np.zeros([3, total])
    for i, pred in enumerate(predls):
        for j, p in enumerate(pred):
            sorted_indices = (-p).argsort()
            top_k_indices = sorted_indices[:n]  
            if true[j] in top_k_indices:
                record[i][j] = 1
    
    analyze_count = [0]*8
    for j in range(total):
        for i in range(8):
            binary = bin(i)[2:]
            check = True
            for k in range(len(binary)):
                if binary[-1-k] == "1" and record[k][j] == 0:
                    check = False
                    break
            if check:
                analyze_count[i] += 1
    return record, analyze_count

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
            predl += pred
            ans = torch.max(pred,1).indices
            preds += ans
            true += y
    predl = torch.stack(predl)
    true = torch.stack(true)
    true = torch.tensor(true).cpu().numpy()
    preds = torch.tensor(preds).cpu().numpy()
    return predl, true


def score_acc(num_moves, data_type, model):
    
    batch_size = 64
    data_size = 30000
    path = 'datas/data_240119.csv'
    data_source = "pros" 
    split_rate = 0.1
    be_top_left = False
    min_move = 0
    max_move = 5
    split = int(num_moves/(max_move-min_move))
    _, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate
                                , be_top_left, train=False)
    print(testData.x.shape)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)
    predl, true = prediction(data_type, model, device, test_loader)
    acc10 = myaccn_split(predl,true,10,split, num_moves)
    acc5 = myaccn_split(predl,true,5,split, num_moves)
    acc1 = myaccn_split(predl,true,1,split, num_moves)
    print(acc10)
    print(acc5)
    print(acc1)
    
def compare_correct(num_moves, device, models, data_types):

    batch_size = 64
    data_size = 3000
    path = 'datas/data_240119.csv'
    data_source = "pros" 
    split_rate = 0.1
    be_top_left = False
    predls = []
    for i, model in enumerate(models):
        _, testData = get_datasets(path, data_types[i], data_source, data_size, num_moves, split_rate
                                , be_top_left, train=False)
        test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)
        true = testData.y
        predl, true = prediction(data_types[i], model, device, test_loader)
        predls.append(predl)
    record1, analyze_count1 = compare_acc(predls, true, 1)
    for i in range(1,8):
        analyze_count1[i] /= analyze_count1[0]
    return record1, analyze_count1


def score_self(num_moves, data_type, model, score_type):
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


def score_more(num_moves, device, model_size, score_type):
    data_types = ['Picture', 'Picture', 'Picture']
    model_names = ["ResNet", "ViT", "ST"]
    states = ['models_80/ResNet1_30000.pt',
              'models_80/ViT1_30000.pt',
              'models_80/ST1_30000.pt']

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
        record, analyze_count = compare_correct(num_moves, device, models, data_types)
        print(analyze_count)

if __name__ == "__main__":
    num_moves = 80
    data_type = "Word"
    model_name = "BERTp"
    model_size = "mid"
    device = "cuda:1"
    state = torch.load('models_80/BERT11p1_140000_35000.pt')
    
    model = get_model(model_name, model_size).to(device)
    model.load_state_dict(state)

    score_acc(num_moves, data_type, model)
    #score_self(num_moves, data_type, model, "score")
    #score_more(num_moves, device, model_size, "correct_compare")
   
    
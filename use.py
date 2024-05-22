import numpy as np
import torch
from tqdm import tqdm

from get_models import get_model
from tools import transfer, transfer_back
from resnet_board import *

def prediction(data_type, model, device, test_loader):
    model.eval()
    predl = []
    true = []
    with torch.no_grad():
        for datas in tqdm(test_loader, leave=False):
            if data_type == "Word":
                x, m, t, y = datas
                x = x.to(device)
                m = m.to(device)
                t = t.to(device)
                y = y.to(device)
                pred = model(x, m, t)
            elif data_type == "Combine":
                xw, m, tt, xp, y = datas
                xw = xw.to(device)
                xp = xp.to(device)
                m = m.to(device)
                tt = tt.to(device)
                y = y.to(device)
                pred = model(xw, m, tt, xp)
            else:
                x, y = datas
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            pred = torch.nn.functional.softmax(pred, dim=-1)
            predl.extend(pred.cpu().numpy())
            true.extend(y.cpu().numpy())

    return predl, true


def next_moves(data_type, num_moves, model, games, num, device):
    games = [[transfer(step) for step in game] for game in games]

    if data_type == 'Word':
        board = get_board(games)[-1]
        last = 0
        while(last < len(games[0])):
            games[0][last] += 1
            last += 1
        games[0].append(362)
        while(len(games[0]) < num_moves):
            games[0].append(0)
        token_types = np.zeros((1, num_moves))
        for j, move in enumerate(games[0]):
            if move == 362:
                break
            move -= 1
            token_types[0][j] = board[move//19][move%19]
        x = torch.tensor(games, dtype=torch.long).to(device)
        mask = (x != 0).detach().long().to(device)
        t = torch.tensor(token_types, dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x, mask, t)[0]
    
    elif data_type == 'LPicture':
        datas = np.zeros([1,4,19,19],  dtype=np.float32)
        for j, move in enumerate(games[0]):
            x = int(move/19)
            y = int(move%19)
            Lchannel_01(datas, 0, x, y, j+1)
            Lchannel_3(datas, 0, x, y, j+1)
        Lchannel_2(datas, 0, len(games[0]))
        x = torch.tensor(datas).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)[0]
    elif data_type == 'Picture':
        datas = np.zeros([1,16,19,19],  dtype=np.float32)
        for j, move in enumerate(games[0]):
            x = int(move/19)
            y = int(move%19)
            channel_01(datas, 0, x, y, j+1)
            channel_1015(datas, 0, x, y, j+1)
        channel_2(datas, 0)
        channel_3(datas, 0, len(games[0]))
        channel_49(datas, 0, len(games[0])-1, games[0])
        x = torch.tensor(datas).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)[0]
    elif data_type == 'Combine':
        
        datas = np.zeros([1,4,19,19],  dtype=np.float32)
        for j, move in enumerate(games[0]):
            x = int(move/19)
            y = int(move%19)
            Lchannel_01(datas, 0, x, y, j+1)
            Lchannel_3(datas, 0, x, y, j+1)
        Lchannel_2(datas, 0, len(games[0]))
        xp = torch.tensor(datas).to(device)

        board = get_board(games)[-1]
        last = 0
        while(last < len(games[0])):
            games[0][last] += 1
            last += 1
        games[0].append(362)
        while(len(games[0]) < num_moves):
            games[0].append(0)
        token_types = np.zeros((1, num_moves))
        for j, move in enumerate(games[0]):
            if move == 362:
                break
            move -= 1
            token_types[0][j] = board[move//19][move%19]
        xw = torch.tensor(games, dtype=torch.long).to(device)
        mask = (xw != 0).detach().long().to(device)
        t = torch.tensor(token_types, dtype=torch.long).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(xw, mask, t, xp)[0]
    if len(games[0]) > 2:
        ban_move = games[0][-2]
        pred[ban_move] = -1e9
    pred = torch.nn.functional.softmax(pred, dim=-1)
    top_indices = np.argsort(pred.cpu().numpy())[-num:]
    return top_indices, torch.tensor([pred[i] for i in top_indices]).numpy()

def vote_next_move(games, device):
    data_config = {}
    data_config["num_moves"] = 240
    model_config = {}
    model_config["model_size"] = "mid"
    anses = []
    probs = []
    
    data_type = 'Word'
    model_config["model_name"] = "BERT"
    model = get_model(model_config).to(device)
    state = torch.load(f'D:\codes\python\.vscode\Language_Go\models\BERTex\mid_s45_20000.pt', map_location=device)
    model.load_state_dict(state)
    ans, prob = next_moves(data_type, data_config["num_moves"], model, games, 10, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)
    
    data_type = 'LPicture'
    model_config["model_name"] = "LResNet"
    model = get_model(model_config).to(device)
    state = torch.load(f'D:\codes\python\.vscode\Language_Go\models\LResNet\mid_s27_20000.pt', map_location=device)
    model.load_state_dict(state)
    ans,prob = next_moves(data_type, data_config["num_moves"], model, games, 10, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)

    vote = {}
    for i, prob in enumerate(probs):
        for j, p in enumerate(prob):
            if anses[i][j] in vote.keys():
                vote[anses[i][j]] += p
            else:
                vote[anses[i][j]] = p
    sorted_vote = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True)[:10])
    results = list(sorted_vote.keys())
    moves = [transfer_back(result[0]*19+result[1]) for result in results]
    return results, moves


if __name__ == "__main__":

    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 35000
    data_config["offset"] = 0
    data_config["data_type"] = "Word"
    data_config["data_source"] = "pros"
    data_config["num_moves"] = 240

    model_config = {}
    model_config["model_name"] = "BERT"
    model_config["model_size"] = "mid"


    device = "cuda:1"
    games = [['dq','dd','pp','pc','qe','co','od','oc','nd','nc','md','lc','mc','mb','cp','do','ld',
              'kc','kd','jc','jd','ic','bo','bn','bp','cm','qc','pd','qd','pe','pf','qf','qg',
              'rf','rg','of','pg','oe','id','hd','he','ge','gd','hc','fd','hf','ie','gf','pb',
              'ob','ee','cf','de','ce','eg','gh','cd','cc','bd','bc','dc','be','ed','ad','qb',
              'jg','dd','dh','eh','di','ei','lg','dj','cj','ck','dk','ej','bk','ci','cl','dg',
              'ch','cg','bh','bg','bi','qq','cb','db','da','ab','ac','af','ae','ea','ca','fb',
              'gb','gc','hb','og','ng','nf','mf','ne','gj','nh','mg','lb','na','df','bb','aa',
              'eq','ep','fq','fp','gp','gq','gr','hq','dr','dp','hr','iq','ir','jq','cr','la',
              'ka','go','jr','kq','kr','lr','lq','mr','lp','mh','nq','nr','oq','or','io','hp',
              'ko','pa','oa','lh','kh','ki','ji','kj','jj','mq','mp','kk','oo','kf','kg','if',
              'ig','qm','pm','ql']]

    anses = []
    probs = []
    
    data_type = 'Word'
    model_config["model_name"] = "BERT"
    model = get_model(model_config).to(device)
    state = torch.load(f'models/BERTex/mid_s45_20000.pt')
    model.load_state_dict(state)
    ans, prob = next_moves(data_type, data_config["num_moves"], model, games, 10, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)
    
    data_type = 'LPicture'
    model_config["model_name"] = "LResNet"
    model = get_model(model_config).to(device)
    state = torch.load(f'models/LResNet/mid_s27_20000.pt')
    model.load_state_dict(state)
    ans,prob = next_moves(data_type, data_config["num_moves"], model, games, 10, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)
    

    data_type = 'Combine'
    model_config["model_name"] = "Combine"
    model = get_model(model_config).to(device)
    state = torch.load(f'models/Combine/B20000_R20000.pt')
    model.load_state_dict(state)
    ans,prob = next_moves(data_type, data_config["num_moves"], model, games, 10, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)

    vote = {}
    for i, prob in enumerate(probs):
        for j, p in enumerate(prob):
            if anses[i][j] in vote.keys():
                vote[anses[i][j]] += p
            else:
                vote[anses[i][j]] = p
    sorted_vote = dict(sorted(vote.items(), key=lambda item: item[1], reverse=True))
    print(sorted_vote)
    
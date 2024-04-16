import numpy as np
import torch
from tqdm import tqdm

from get_models import get_model
from tools import *

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
            else:
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


def next_moves(data_type, num_moves, model, games, num, device):
    games = [[transfer(step) for step in game] for game in games]

    if data_type == 'Word':
        last = 0
        while(last < len(games[0])):
            games[0][last] += 1
            last += 1
        games[0].append(362)
        while(len(games[0]) < num_moves):
            games[0].append(0)
        x = torch.tensor(games).to(device)
        mask = (x != 0).detach().long().to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x, mask)[0]

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
    pred = torch.nn.functional.softmax(pred)
    top_indices = np.argsort(pred.cpu().numpy())[-num:]
    return top_indices, torch.tensor([pred[i] for i in top_indices]).numpy()



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
    model_config["config_path"] = "models_160/p1/config.json"
    model_config["state_path"] = "models_160/p1/model.safetensors"

    device = "cuda:0"
    games = [['dq','dd','pp','pd','cn','nq','qn','jp','qc','qd','pc','nc','rd','re','od','oc','qe',
              'pe','qf','rc','pf','oe','of','ne','nf','me','rf','sd','cc','dc','oq','np','cd',
              'cf','db','eb','cb','fc','de','ce','ee','fd','cg','df','ef','dg','eg','dh','ed',
              'ec','hc','hd','id','he','ie','hf','eh','di','ei','dj','ej','dk','if','hg','gc',
              'gd','nr','mr','or','fq','be','bf','ig','hh','ek','dl','hq','hp','iq','ip','gq',
              'gp','fr','jq','el','dm','gb','fb','fp','fo','eq','dn','co','fm','ih','hi','ii',
              'hj','bd','ij','ic','kc','mf','lf','lg','kf','kg','pm','qm','pl','ql']]

    anses = []
    probs = []
    data_type = 'Word'
    model_config["model_name"] = "BERT"
    model = get_model(model_config).to(device)
    state = torch.load(f'models/BERT/mid_s2_7500x4.pt')
    model.load_state_dict(state)
    ans,prob = next_moves(data_type, data_config["num_moves"], model, games, 5, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)

    data_type = 'Word'
    model_config["model_name"] = "BERTp"
    model = get_model(model_config).to(device)
    state = torch.load(f'models/BERT/mid_s27_30000.pt')
    model.load_state_dict(state)
    ans,prob = next_moves(data_type, data_config["num_moves"], model, games, 5, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    anses.append(ans)
    probs.append(prob)

    data_type = 'Picture'
    model_config["model_name"] = "ResNet"
    model = get_model(model_config).to(device)
    state = torch.load(f'models/ResNet/mid_10000.pt')
    model.load_state_dict(state)
    ans,prob = next_moves(data_type, data_config["num_moves"], model, games, 5, device)
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
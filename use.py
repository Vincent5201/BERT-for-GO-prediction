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
        games = np.insert(games, 0, 363, axis=1)
        x = torch.tensor(games).to(device)
        mask = (x != 0).detach().long().to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x, mask)[0]
    elif data_type == 'Word':
        last = 0
        while(last < len(games[0])):
            games[0][last] += 1
            last += 1
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
           
    top_indices = np.argsort(pred.cpu().numpy())[-num:]
    return top_indices, torch.tensor([pred[i] for i in top_indices])



if __name__ == "__main__":

    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 30000
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
    games = [['dq','dd','pd','pp','co','qc','qd','pc','nc','oc','od','nb','rc','rb','mb','mc',
              'nd','lb','ob','rd','qb','sc','pb','oo','jq','ma','rc','kc','re','ce','cj','qc',
              'ra','iq','ip','hq','jr','hp','io','ho','in','hn','hm','im','gm','il','ko','qj',
              'kl','ij','hk','ik','hj','hi','gi','hh','gh','hg','gg','gf','ff','ge','fe','fd',
              'gd','hd','gc','fc','hc','ic','he','hf','id','jd','je','hl','gl','ke','jf','gb',
              'hb','fb','kf','ib','le','gj','gk','fj','eh','fk','el','di','dh','ci','ch','bi',
              'bh','fl','fm','em','en','dm','fn','dl','ii','ha','ji','ei','df','hd','bf','dn',
              'do','eo','fo','ep','fp','eq','fq','er','fr','dr','cq','cr','br','bn','bo','cn',
              'cs','pi','qq','qp','pq','oq','or','nr','nq','op','mr','rq','rr','pr','ns','qr',
              'qh','be','ai','bj','cf','bq','bp','ph','qg','bs','as','kk','ri','ll','nl','lj',
              'pg','ql','km','ki','jj','jk','kh','lh','jh','lm','ln','mn','mo','mm','no','lg',
              'kd','ie','nn','nm','on','om','pn','kg','jg','qn','pm','pl']]


    data_type = 'BERT'
    model_config["model_name"] = "BERT"
    model = get_model(model_config).to(device)
    state = torch.load(f'models_{data_config["num_moves"]}/BERT1_s27_30000.pt')
    model.load_state_dict(state)
    ans,_ = next_moves(data_type, data_config["num_moves"], model, games, 3, device)
    ans = [(int(step/19),int(step%19)) for step in ans]
    print(ans)
    

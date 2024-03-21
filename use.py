import numpy as np
import torch
from tqdm import tqdm

from myDatasets import transfer, channel_01, channel_1015, channel_2, channel_3, channel_49
from myModels import get_model

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
            channel_01(datas, 0, x, y, j)
            channel_1015(datas, 0, x, y, j)
        channel_2(datas, 0)
        channel_3(datas, 0, len(games[0])-1)
        channel_49(datas, 0, len(games[0])-1, games[0])
        x = torch.tensor(datas).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(x)[0]
           

    top_indices = np.argsort(pred.cpu().numpy())[-num:]
    return top_indices, torch.tensor([pred[i] for i in top_indices])

if __name__ == "__main__":
    data_type = 'Picture'
    num_moves = 80
    model = get_model("ST", "mid").to("cuda:1")
    state = torch.load('models_80/ST1_30000.pt')
    model.load_state_dict(state)
    games = [['cd','dq','qd','pp','co','fp','oc','ec','df','hc','nq','qn']]
    
    ans,_ = next_moves(data_type, num_moves, model, games, 5)
    ans = [(int(step/19),int(step%19)) for step in ans]
    print(ans)

    data_type = 'Picture'
    num_moves = 80
    model = get_model("ViT", "mid").to("cuda:1")
    state = torch.load('models_80/ViT1_30000.pt')
    model.load_state_dict(state)
    ans,_ = next_moves(data_type, num_moves, model, games, 5)
    ans = [(int(step/19),int(step%19)) for step in ans]
    print(ans)

    data_type = 'Picture'
    num_moves = 80
    model = get_model("ResNet", "mid").to("cuda:1")
    state = torch.load('models_80/ResNet1_30000.pt')
    model.load_state_dict(state)
    ans,_ = next_moves(data_type, num_moves, model, games, 5)
    ans = [(int(step/19),int(step%19)) for step in ans]
    print(ans)

    data_type = 'Word'
    num_moves = 80
    model = get_model("BERT", "mid").to("cuda:1")
    state = torch.load('models_80/BERT1.pt')
    model.load_state_dict(state)
    ans,_ = next_moves(data_type, num_moves, model, games, 5)
    ans = [(int(step/19),int(step%19)) for step in ans]
    print(ans)
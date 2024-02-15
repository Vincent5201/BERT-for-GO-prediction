import numpy as np
import torch

from myDatasets import transfer, channel_01, channel_1015, channel_2, channel_3, channel_49
from myModels import get_model


def next_moves(data_type, num_moves, model, games, num):
    games = [[transfer(step) for step in game] for game in games]

    if data_type == 'Word':
        last = 0
        while(last < len(games[0])):
            games[0][last] += 1
            last += 1
        while(len(games[0]) < num_moves):
            games[0].append(0)
        x = torch.tensor(games).to("cuda:1")
        mask = (x != 0).detach().long().to("cuda:1")

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
        x = torch.tensor(datas).to("cuda:1")

        model.eval()
        with torch.no_grad():
            pred = model(x)[0]
           

    top_indices = np.argsort(pred.cpu().numpy())[-num:]
    return top_indices, torch.tensor([pred[i] for i in top_indices])

if __name__ == "__main__":
    data_type = 'Picture'
    num_moves = 80
    model = get_model("ResNet", "mid").to("cuda:1")
    state = torch.load('models/ResNet1.pt')
    model.load_state_dict(state)
    games = [["dd",'dp','pc','qp','oq','po','pe','lq','cq','dq','cp','do','bn','np','cm','cc','cd',
              'dc','fc','ec','ed','fb','qq','rq','rr','qr','pq','sr','gc','gb','hc','qg','qi',
              'qd','qe','iq','hb','pi','pj']]
    
    ans,_ = next_moves(data_type, num_moves, model, games, 5)
    ans = [(int(step/19),int(step%19)) for step in ans]
    print(ans)
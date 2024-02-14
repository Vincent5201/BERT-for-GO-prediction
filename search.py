import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from value import spread_value
from myDatasets import transfer, transfer_back
from myModels import get_model



class Beam():
    def __init__(self, game, max_len):
        self.game_ch = copy.deepcopy(game)
        self.game = copy.deepcopy(game)
        self.game = [transfer(move) for move in self.game]
        self.num = len(self.game)
        for i in range(self.num):
            self.game[i] += 1
        while(len(self.game) < max_len):
            self.game.append(0)
        self.score = spread_value(self.game_ch)

    def add_step(self, step):
        if step in self.game_ch:
            self.score = None
        else:
            self.game[self.num] = step + 1
            self.num += 1
            self.game_ch.append(transfer_back(step))
            self.score = spread_value(self.game_ch)


def value_search(games, num_beam, num_output, num_output_moves, max_len, data_type, model):
    
    inp_len = len(games[0])

    beams = [Beam(games[0], max_len)]
    for i in tqdm(range(num_output_moves), total=num_output_moves):
        new_beams = []
        for beam in beams:
            with torch.no_grad():
                x = torch.tensor(beam.game).unsqueeze(0).to("cuda:0")
                if data_type == "Word": 
                    mask = (x != 0).detach().long().to("cuda:0")
                    pred = model(x, mask)[0]
                else:
                    # convert x to picture
                    pred = model(x.float())[0]
            probs = F.softmax(pred,dim=0).cpu().numpy()
            top_indices = np.argsort(probs)
            top_indices = np.flip(top_indices)[:num_beam]
            for i in range(num_beam):
                new_beam = Beam(beam.game_ch, max_len)
                new_beam.add_step(top_indices[i])
                new_beams.append(new_beam)
        new_beams = [beam for beam in new_beams if beam.score != None]
        new_beams = sorted(new_beams, key=lambda obj: obj.score)
        if (inp_len+i) % 2:
            beams = copy.deepcopy(new_beams[:num_beam])
        else:
            beams = copy.deepcopy(new_beams[-num_beam:])
            beams.reverse()
    ans = [beam.game_ch[inp_len:] for beam in beams[:num_output]]
    
    return ans
    

if __name__ == "__main__":
    model = get_model("BERT", "mid").to("cuda:0")
    state = torch.load('/home/F74106165/Transformer_Go/models/BERT1.pt')
    model.load_state_dict(state)
    model.eval()

    games = [["cd",'dq','pq','qd','oc','qo','co','ec', 'de','pe','np','fp']]
    ans = value_search(games, 10, 3, 5, 80, "Word", model)
    print(ans)

import copy
import torch
from tqdm import tqdm
import torch.nn.functional as F

from value import spread_value
from myDatasets import transfer, transfer_back
from myModels import get_model
from use import next_moves



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
        self.score = 0

    def add_step(self, step, prob):

        if transfer_back(step) in self.game_ch:
            self.score = None
        else:
            self.game[self.num] = step + 1
            self.num += 1
            self.game_ch.append(transfer_back(step))
            self.score += float(prob)


def value_search(games, num_beam, num_output, num_output_moves, max_len, data_type, model):
    inp_len = len(games[0])
    beams = [Beam(games[0], max_len)]
    for i in tqdm(range(num_output_moves), total=num_output_moves):
        new_beams = []
        for beam in beams:
            top_indices, probs = next_moves(data_type, max_len, model, [beam.game_ch], num_beam)
            probs = F.softmax(probs,dim=-1)
            for i in range(num_beam):
                new_beam = Beam(beam.game_ch, max_len)
                new_beam.add_step(top_indices[i], probs[i])
                new_beams.append(new_beam)
        new_beams = [beam for beam in new_beams if beam.score != None]
        new_beams = sorted(new_beams, key=lambda obj: obj.score)
        beams = copy.deepcopy(new_beams[-num_beam:])
        beams.reverse()
        if inp_len+i == max_len:
            break
    ans = [beam.game_ch[inp_len:] for beam in beams[:num_output]]
    score = [beam.score for beam in beams[:num_output]]
    
    return ans, score
    

if __name__ == "__main__":
    model = get_model("ResNet", "mid").to("cuda:1")
    state = torch.load('/home/F74106165/Transformer_Go/models/ResNet1.pt')
    model.load_state_dict(state)
    model.eval()

    games = [["dd",'dp','pc','qp','oq','po','pe','lq','cq','dq','cp','do','bn','np','cm','cc','cd',
              'dc','fc','ec','ed','fb','qq','rq','rr','qr','pq','sr','gc','gb','hc','qg','qi',
              'qd','qe','iq','hb','pi','pj']]
    ans, score = value_search(games, 10, 5, 10, 80, "Picture", model)
    print(ans)
    print(score)

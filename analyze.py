import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from get_datasets import get_datasets
from get_models import get_model

def cosine_similarity(vec1, vec2):
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    if magnitude_vec1 != 0 and magnitude_vec2 != 0:
        similarity = np.dot(vec1, vec2) / (magnitude_vec1 * magnitude_vec2)
    else:
        similarity = 0
    return similarity

def embedding_distance(data_config, model_config, model_path):
    if data_config["data_type"] != "Word":
        print("wrong data type")
        return
    _, testData = get_datasets(data_config, 1, train=False)
    games = torch.stack([testData.x[data_config["num_moves"]*i+data_config["num_moves"]-1]\
                          for i in range(int(len(testData.x)/data_config["num_moves"]))])
    print(games.shape)
    model = get_model(model_config)
    state = torch.load(model_path)
    model.load_state_dict(state)

    mat = np.zeros((361,361))
    count = np.zeros((361,361))
    model.eval()
    embedding_weights = model.bert.get_input_embeddings()
    input_embeddings = embedding_weights(games).detach().numpy()
    for i, (game, game_v) in tqdm(enumerate(zip(games, input_embeddings)), total=len(games), leave=False):
        for j, (move, move_v) in enumerate(zip(game, game_v)):
            if move and move < 362:
                move -= 1
                for k, (move2, move_v2) in enumerate(zip(game, game_v)):
                    if k > j and move2 and move2 < 362 and move != move2:
                        move2 -= 1
                        simi = cosine_similarity(move_v, move_v2)
                        mat[move][move2] += simi
                        count[move][move2] += 1
                        mat[move2][move] += simi
                        count[move2][move] += 1
    for i in range(361):
        for j in range(361):
            if count[i][j]:
                mat[i][j] /= count[i][j]
    np.save('analyzation_data/cos_simi_tmp.npy', mat)

    return mat

def data_similarity(data_config):
    _, testData = get_datasets(data_config, 1, train=False)
    games = torch.stack([testData.x[data_config["num_moves"]*i+data_config["num_moves"]-1]\
                        for i in range(int(len(testData.x)/data_config["num_moves"]))]).cpu().numpy()
    print(games.shape)
    counts = [0]*(data_config["num_moves"]+1)
    records = np.zeros((len(games), 361))
    for i, game in tqdm(enumerate(games), total=len(games), leave=False):
        for p in range(19):
            for q in range(19):
                if game[0][p][q]:
                    records[i][19*p+q] = 1
                elif game[1][p][q]:
                    records[i][19*p+q] = -1
    print("records end")
    for i, record1 in tqdm(enumerate(records), total=len(records), leave=False):
        for j, record2 in enumerate(records):
            if j > i:
                counts[np.sum((record1 != 0) & (record1 == record2))] += 1
    print(counts)
    return counts

def check_atari(game, x, y, p):
    pp = 1
    if p:
        pp = 0
    count = 0
    if x > 0 and x < 18 and y > 0 and y < 18:
        if game[p][x-1][y] or game[p][x+1][y] or game[p][x][y-1] or game[p][x][y+1]:
            return -1
        if game[pp][x-1][y+1] or game[pp][x+1][y-1] or game[pp][x-1][y-1] or game[pp][x+1][y+1]:
            return -1

        if game[pp][x-1][y]:
            count += 1
        if game[pp][x+1][y]:
            count += 1
        if game[pp][x][y-1]:
            count += 1
        if game[pp][x][y+1]:
            count += 1
        if count == 3:
            return x*19+y
    return -1
    
def plot_board(mat):
    mat = np.array(mat).reshape(19,19)
    cmap = plt.get_cmap('viridis')
    plt.imshow(mat, cmap=cmap)
    plt.colorbar()
    plt.show()

def find_atari(games, trues):
    pos = [0]*361
    games = games.cpu().numpy()
    for i, game in tqdm(enumerate(games), total=len(games), leave=False):
        x = int(trues[i]/19)
        y = int(trues[i]%19)
        if x > 0 and game[i%2][x-1][y] and game[10][x-1][y]:
            ret = check_atari(game, x-1, y, i%2)
            if ret != -1:
                pos[ret] += 1
        if x < 18 and game[i%2][x+1][y] and game[10][x+1][y]:
            ret = check_atari(game, x+1, y, i%2)
            if ret != -1:
                pos[ret] += 1
        if y > 0 and game[i%2][x][y-1] and game[10][x][y-1]:
            ret = check_atari(game, x, y-1, i%2)
            if ret != -1:
                pos[ret] += 1
        if y < 18 and game[i%2][x][y+1] and game[10][x][y+1]:
            ret = check_atari(game, x, y+1, i%2)
            if ret != -1:
                pos[ret] += 1

    plot_board(pos)
    return

def plot_moves(counts):
    counts = np.array(counts)
    plt.bar([i+1 for i in range(len(counts))], np.log2(counts))
    plt.title('Sample Bar Chart')
    plt.xlabel('number of same stones')
    plt.ylabel('counts')
    plt.show()

if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 30000
    data_config["offset"] = 0
    data_config["data_type"] = "Word"
    data_config["data_source"] = "pros"
    data_config["num_moves"] = 80

    model_config = {}
    model_config["model_name"] = "BERT"
    model_config["model_size"] = "mid"
    model_config["config_path"] = "models_160/p1/config.json"
    model_config["state_path"] = "models_160/p1/model.safetensors"

    device = "cuda:1"
    model_path = f'models/BERT/mid_s2_7500x4.pt'
    #data_similarity(data_config)
    #mats = embedding_distance(data_config, model_config, model_path)

    counts = [0, 0, 0, 0, 3, 5, 4, 9, 18, 29, 44, 82, 108, 180, 300, 413, 634, 929, 1301, 1847, 2537, 3517, 4734, 6278, 8093, 10494, 13072, 16367, 20027, 24224, 29012, 34322, 40202, 46248, 52605, 59746, 66213, 73572, 81609, 89042, 96171, 104141, 111140, 117624, 124894, 130395, 136040, 141311, 146133, 151037, 154219, 156925, 158753, 160693, 161605, 161574, 161688, 160282, 158278, 155757, 152850, 149517, 145254, 141371, 135921, 131139, 124606, 118970, 112843, 105863, 98829, 91893, 85234, 77992, 70196, 64053, 57559, 51640, 44759, 38585, 33500, 28824, 24123, 20362, 16842, 13580, 11125, 8717, 6870, 5469, 4219, 3056, 2364, 1772, 1254, 952, 633, 456, 317, 215, 157, 116, 63, 48, 25, 16, 9, 5, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    plot_moves(counts)


   
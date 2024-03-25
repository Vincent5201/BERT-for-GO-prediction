import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from myDatasets import get_datasets
from myModels import get_model

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    if magnitude_vec1 != 0 and magnitude_vec2 != 0:
        similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    else:
        similarity = 0
    return similarity

def embedding_distance(model, games):
    mat = np.zeros((361,361))
    count = np.zeros((361,361))
    model.eval()
    embedding_weights = model.bert.get_input_embeddings()
    input_embeddings = embedding_weights(games).detach().numpy()
    for i, (game, game_v) in tqdm(enumerate(zip(games, input_embeddings)), total=len(games), leave=False):
        for j, (move, move_v) in enumerate(zip(game, game_v)):
            if move > 0 and move < 362:
                for k, (move2, move_v2) in enumerate(zip(game, game_v)):
                    if k > j and move2 > 0 and move2 < 362:
                        move -= 1
                        move2 -= 1
                        dis = cosine_similarity(move_v, move_v2)
                        mat[move][move2] += dis
                        count[move][move2] += 1
                        mat[move2][move] += dis
                        count[move2][move] += 1
    for i in range(361):
        for j in range(361):
            if count[i][j]:
                mat[i][j] /= count[i][j]
    np.save('dis.npy', mat)

    return mat
    

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
   
    trainData, testData = get_datasets(data_config, 1, train=False)
    games = torch.stack([testData.x[80*i+79] for i in range(int(len(testData.x)/80))])
    print(games.shape)
    model = get_model(model_config)
    mats = embedding_distance(model, games)
    
    mat = np.load('D:/codes/python/.vscode/Transformer_Go/tmp.npy')
    plot_board(mat[68])
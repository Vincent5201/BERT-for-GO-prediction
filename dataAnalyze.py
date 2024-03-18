import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import copy

from myDatasets import get_datasets


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
    cmap = plt.get_cmap('coolwarm')
    plt.imshow(mat, cmap=cmap)
    cbar = plt.colorbar()
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

def rotate_matrix_90(matrix):
    n = 19
    rotated_matrix = [0] * n * n
    for i in range(n):
        for j in range(n):
            rotated_i = j
            rotated_j = n - 1 - i
            rotated_matrix[rotated_i * n + rotated_j] = matrix[i * n + j]
    
    return rotated_matrix

if __name__ == "__main__":
    
    path = 'datas/data_240119.csv'
    data_source = "pros"
    data_type = 'Picture'
    num_moves = 80
    data_size = 30000
    split_rate = 1
    be_top_left = False
    #_, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate, be_top_left, train=False)
    
    #find_atari(testData.x, testData.y)
    with open('D:/codes/python/.vscode/Transformer_Go/analyzation.yaml', 'r') as file:
        args = yaml.safe_load(file)
    mat1 = args["pos_recall"]["model_240"]["ResNet"]
    mat2 = args["pos_recall"]["model_240"]["ST"]
    mat3 = args["pos_recall"]["model_240"]["ViT"]

    tmp = copy.deepcopy(mat1)
    for _ in range(3):
        tmp = rotate_matrix_90(tmp)
        for i in range(361):
            mat1[i] += tmp[i]
    for i in range(361):
        mat1[i] /= 4
    
    tmp = copy.deepcopy(mat2)
    for _ in range(3):
        tmp = rotate_matrix_90(tmp)
        for i in range(361):
            mat2[i] += tmp[i]
    for i in range(361):
        mat2[i] /= 4
    
    tmp = copy.deepcopy(mat3)
    for _ in range(3):
        tmp = rotate_matrix_90(tmp)
        for i in range(361):
            mat3[i] += tmp[i]
    for i in range(361):
        mat3[i] /= 4
    

    mat12 = [mat1[i]-mat2[i] for i in range(361)]
    mat13 = [mat1[i]-mat3[i] for i in range(361)]
    mat23 = [mat2[i]-mat3[i] for i in range(361)]
    plot_board(mat23)
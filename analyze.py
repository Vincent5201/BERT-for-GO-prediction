import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns  

from get_datasets import get_datasets
from get_models import get_model

def draw_confusion_matrix():

    predls = np.load('analyze_data/predls3_20000.npy')
    true = np.load('analyze_data/trues3.npy')
    predl = torch.tensor(predls[0])
    preds = torch.max(predl,1).indices
    
    cm = confusion_matrix(true, preds)
    np.save('analyze_data/cmR.npy', cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues', xticklabels=[0,1,2,3,4,5,6,7,8,9], yticklabels=[0,1,2,3,4,5,6,7,8,9])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

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
    _, testData = get_datasets(data_config, train=False)
    games = torch.stack([testData.x[data_config["num_moves"]*i+data_config["num_moves"]-1]\
                          for i in range(int(len(testData.x)/data_config["num_moves"]))])

    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

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
    for i, cx in enumerate(count):
        for j, cy in enumerate(cx):
            if cy:
                mat[i][j] /= cy
    np.save('analyze_data/cos_simi_tmp.npy', mat)
    
    return mat

def embedding_distance(data_config, model_config, model_path):
    if data_config["data_type"] != "Word":
        print("wrong data type")
        return
    _, testData = get_datasets(data_config, train=False)
    games = torch.stack([testData.x[data_config["num_moves"]*i+data_config["num_moves"]-1]\
                          for i in range(int(len(testData.x)/data_config["num_moves"]))])

    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    mat = np.zeros((361,128))
    count = [0]*361
    model.eval()
    embedding_weights = model.bert.get_input_embeddings()
    input_embeddings = embedding_weights(games).detach().numpy()

    for i, (game, game_v) in tqdm(enumerate(zip(games, input_embeddings)), total=len(games), leave=False):
        for j, (move, move_v) in enumerate(zip(game, game_v)):
            if move and move < 362:
                move -= 1
                mat[move] += move_v
                count[move] += 1
            
    for i, c in enumerate(count):
        if c:
            mat[i] /= c
    np.save('analyze_data/embeddings.npy', mat)
    
    return mat


def data_similarity(data_config):
    _, testData = get_datasets(data_config, train=False)
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

def analyze_correct():
    predls = np.load('analyze_data/predls_2.npy')
    trues = np.load('analyze_data/trues.npy')
    total = len(trues)
    models = len(predls)
    n = 1
    nummoves = 240
    correct_moves = [[0]*30 for _ in range(models)]
    for i in tqdm(range(total), total=total, leave=False):
        if i%nummoves:
            for j, predl in enumerate(predls):
                chooses = (-predl[i]).argsort()[:n]
                if trues[i] in chooses:
                    xl = int(trues[i-1]/19)
                    yl = int(trues[i-1]%19)
                    x = int(trues[i]/19)
                    y = int(trues[i]%19)
                    dis = (pow(x-xl, 2) + pow(y-yl, 2)) ** (0.5)
                    correct_moves[j][int(dis+0.5)] += 1
    print(correct_moves)

def labels_recall():
    predls = np.load('analyze_data/predls4.npy')
    trues = np.load('analyze_data/trues4.npy')
    predl = torch.tensor(predls[2])
    preds = torch.max(predl,1).indices
    count = [0]*361
    correct = [0]*361
    for i, true in enumerate(trues):
        count[true] += 1
        if preds[i] == true:
            correct[true] += 1

    ret = [correct[i]/count[i] for i in range(361)]
    np.save('analyze_data/label_recall.npy', ret)

def labels_precision():
    predls = np.load('analyze_data/predls4.npy')
    trues = np.load('analyze_data/trues4.npy')
    predl = torch.tensor(predls[3])
    preds = torch.max(predl,1).indices
    count = [0]*361
    correct = [0]*361
    for i, pred in enumerate(preds):
        count[pred] += 1
        if trues[i] == pred:
            correct[pred] += 1

    ret = [correct[i]/count[i] for i in range(361)]
    np.save('analyze_data/label_precision.npy', ret)

def RB_test():
    predls = np.load('analyze_data/predls4.npy')
    trues = np.load('analyze_data/trues4.npy')
    predlR = torch.tensor(predls[2])
    predlB = torch.tensor(predls[3])

    R_correct = []
    B_correct = []
    for i, (b, r) in tqdm(enumerate(zip(predlB, predlR)), total=len(predlR), leave=False):
        sorted_b = (-b).argsort()
        sorted_r = (-r).argsort()
        top_k_b = sorted_b[:5] 
        top_k_r = sorted_r[:5] 
        if trues[i] in top_k_b:
            B_correct.append(True)
        else:
            B_correct.append(False)
        if trues[i] in top_k_r:
            R_correct.append(True)
        else:
            R_correct.append(False)

    np.save('analyze_data/B5_correct.npy', B_correct)
    np.save('analyze_data/R5_correct.npy', R_correct)

if __name__ == "__main__":
    data_config = {}
    data_config["path"] = 'datas/data_240119.csv'
    data_config["data_size"] = 30000
    data_config["offset"] = 0
    data_config["data_type"] = "Word"
    data_config["data_source"] = "pros"
    data_config["num_moves"] = 240
    data_config["extend"] = False

    model_config = {}
    model_config["model_name"] = "BERT"
    model_config["model_size"] = "mid"
   

    device = "cuda:0"
    model_path = f'models/BERTex/mid_s45_20000.pt'
    #data_similarity(data_config)
    #mats = embedding_distance(data_config, model_config, model_path)
    #analyze_correct()
    #draw_confusion_matrix()
    #labels_precision()
    #labels_recall()
    RB_test()
    



   
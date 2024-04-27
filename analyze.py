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
    for i in range(361):
        for j in range(361):
            if count[i][j]:
                mat[i][j] /= count[i][j]
    np.save('analyzation_data/cos_simi_tmp.npy', mat)
    return mat

def check_atari(game, x, y, p):
    pp = 1
    if p:
        pp = 0
    count = 0
    directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    directions2 = [(x-1, y+1), (x+1, y-1), (x-1, y-1), (x+1, y+1)]
    if x > 0 and x < 18 and y > 0 and y < 18:
        for d in directions2:
            if game[pp][d[0]][d[1]]:
                return -1
        for d in directions:
            if game[p][d[0]][d[1]]:
                return -1
            if game[pp][d[0]][d[1]]:
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
        directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for d in directions:
            if d[0] < 0 or d[1] > 18:
                break
            if game[i%2][d[0]][d[1]] and game[10][d[0]][d[1]]:
                ret = check_atari(game, d[0], d[1], i%2)
                if ret != -1:
                    pos[ret] += 1
    plot_board(pos)
    return

def plot_moves(counts):
    counts = np.array(counts)
    plt.bar([i+1 for i in range(len(counts))], counts)
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

    counts = [1973, 1146, 1422, 1842, 1088, 1633, 1637, 1873, 1805, 1878, 1934, 1911, 1865, 1849, 1852, 1842, 1803, 1795, 1773, 1706, 1713, 1651, 1653, 1637, 1628, 1579, 1605, 1566, 1617, 1528, 1513, 1522, 1492, 1426, 1517, 1486, 1515, 1401, 1445, 1519, 1476, 1479, 1482, 1364, 1449, 1415, 1427, 1360, 1443, 1401, 1369, 1410, 1382, 1398, 1346, 1358, 1427, 1439, 1398, 1396, 1412, 1364, 1406, 1374, 1408, 1345, 1346, 1402, 1421, 1336, 1367, 1300, 1358, 1384, 1366, 1341, 1353, 1387, 1312, 1320, 1350, 1332, 1319, 1318, 1343, 1318, 1326, 1287, 1359, 1322, 1355, 1318, 1330, 1351, 1339, 1357, 1346, 1327, 1322, 1325, 1338, 1317, 1334, 1297, 1308, 1332, 1284, 1271, 1280, 1314, 1349, 1350, 1377, 1338, 1395, 1309, 1323, 1310, 1395, 1311, 1382, 1366, 1348, 1392, 1333, 1310, 1358, 1346, 1426, 1375, 1398, 1374, 1368, 1390, 1426, 1404, 1422, 1381, 1407, 1454, 1434, 1408, 1472, 1445, 1437, 1454, 1476, 1445, 1547, 1495, 1534, 1538, 1532, 1514, 1600, 1573, 1587, 1557, 1620, 1642, 1634, 1661, 1654, 1660, 1722, 1648, 1742, 1700, 1777, 1712, 1729, 1747, 1809, 1765, 1841, 1829, 1872, 1882, 1840, 1861, 1865, 1901, 1925, 1936, 1986, 1988, 2021, 1987, 2056, 2087, 2102, 2059, 2132, 2102, 2148, 2067, 2141, 2181, 2170, 2155, 2222, 2239, 2246, 2216, 2236, 2287, 2292, 2332, 2366, 2285, 2393, 2384, 2416, 2402, 2387, 2402, 2426, 2452, 2471, 2505, 2483, 2503, 2510, 2546, 2539, 2545, 2581, 2525, 2613, 2604, 2637, 2622, 2622, 2640, 2670, 2665, 2696, 2682, 2736, 2732]
    plot_moves(counts)
    counts_2 = [0, 1443, 240, 879, 701, 1152, 1497, 1801, 1767, 1875, 1970, 2061, 1997, 2067, 1961, 2039, 1978, 1941, 1937, 1889, 1821, 1833, 1790, 1768, 1794, 1702, 1727, 1667, 1691, 1655, 1625, 1599, 1538, 1583, 1593, 1535, 1585, 1490, 1447, 1529, 1514, 1530, 1483, 1415, 1437, 1451, 1409, 1416, 1433, 1412, 1421, 1368, 1388, 1413, 1346, 1425, 1392, 1400, 1406, 1390, 1383, 1402, 1394, 1350, 1335, 1336, 1316, 1422, 1351, 1324, 1345, 1327, 1319, 1352, 1294, 1348, 1344, 1358, 1303, 1287, 1301, 1320, 1310, 1287, 1275, 1327, 1280, 1286, 1335, 1315, 1285, 1261, 1289, 1314, 1274, 1328, 1291, 1269, 1299, 1261, 1257, 1279, 1297, 1264, 1259, 1294, 1261, 1236, 1252, 1274, 1304, 1299, 1318, 1275, 1330, 1278, 1270, 1264, 1317, 1257, 1303, 1321, 1288, 1397, 1287, 1307, 1319, 1298, 1353, 1292, 1358, 1312, 1337, 1332, 1403, 1362, 1375, 1369, 1326, 1412, 1402, 1383, 1440, 1427, 1410, 1448, 1403, 1466, 1482, 1477, 1499, 1501, 1468, 1438, 1578, 1581, 1555, 1552, 1541, 1623, 1629, 1622, 1641, 1601, 1643, 1665, 1721, 1741, 1750, 1751, 1764, 1762, 1777, 1790, 1830, 1825, 1864, 1894, 1838, 1867, 1875, 1887, 1911, 1963, 2015, 1993, 2016, 1991, 2052, 2020, 2080, 2079, 2132, 2102, 2153, 2107, 2151, 2175, 2158, 2187, 2231, 2243, 2236, 2241, 2242, 2279, 2261, 2340, 2346, 2307, 2382, 2388, 2430, 2405, 2419, 2409, 2447, 2448, 2477, 2489, 2498, 2513, 2529, 2563, 2554, 2576, 2590, 2571, 2619, 2624, 2649, 2645, 2648, 2659, 2712, 2669, 2724, 2711, 2752, 2750]
    plot_moves(counts_2)

   
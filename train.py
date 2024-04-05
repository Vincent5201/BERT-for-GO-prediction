import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random

from get_datasets import get_datasets
from get_models import get_model
from tools import myaccn

data_config = {}
data_config["path"] = 'datas/data_240119.csv'
data_config["data_size"] = 10000
data_config["offset"] = 0
data_config["data_type"] = "BERT"
data_config["data_source"] = "pros"
data_config["num_moves"] = 240

model_config = {}
model_config["model_name"] = "BERTCNN"
model_config["model_size"] = "mid"
model_config["config_path"] = "models_160/p1/config.json"
model_config["state_path"] = "models_160/p1/model.safetensors"

batch_size = 64
num_epochs = 50
lr = 5e-5
device = "cuda:0"
save = True
random_seed = random.randint(0,100)
print(f'rand_seed:{random_seed}')

random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
model = get_model(model_config).to(device)

trainData, testData = get_datasets(data_config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fct = nn.CrossEntropyLoss()

train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testData, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    losses = []
    for datas in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        if data_config["data_type"] == "BERT":
            x, m, y = datas
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            pred = model(x, m)
        else:
            x, y = datas
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
        loss = loss_fct(pred,y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    print(f'epoch {epoch+1}/{num_epochs},  train_loss = {sum(losses)/len(losses):.4f}')

    model.eval()
    preds = []
    predl = []
    true = []
    with torch.no_grad():
        for datas in tqdm(test_loader, leave=False):
            if data_config["data_type"] == "BERT":
                x, m, y = datas
                x = x.to(device)
                m = m.to(device)
                y = y.to(device)
                pred = model(x, m)
            else:
                x, y = datas
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            predl += pred
            ans = torch.max(pred,1).indices
            preds += ans
            true += y
    predl = torch.stack(predl)
    true = torch.stack(true)
    print(f'val_loss:{loss_fct(pred,y):.4f}')

    true = torch.tensor(true).cpu().numpy()
    preds = torch.tensor(preds).cpu().numpy()
    print(f'accuracy5:{myaccn(predl,true, 5)}')
    print(f'accuracy:{accuracy_score(preds,true)}')
    if save:
        torch.save(model.state_dict(), f'/home/F74106165/Transformer_Go/tmpmodels/model{epoch+1}.pt')
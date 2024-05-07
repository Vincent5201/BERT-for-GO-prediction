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
data_config["data_size"] = 5000
data_config["offset"] = 0
data_config["data_type"] = "LPicture"
data_config["data_source"] = "pros"
data_config["num_moves"] = 240
data_config["extend"] = False

model_config = {}
model_config["model_name"] = "LResNet"
model_config["model_size"] = "mid"

batch_size = 64
num_epochs = 50
lr = 5e-4
device = "cuda:0"
save = True
random_seed = random.randint(0,100)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
print(f'rand_seed:{random_seed}')

path2 = "models/BERTex/mid_s63_5000.pt"
path1 = "models/ResNet/mid_s57_5000.pt"
path1 = path2 = None
model = get_model(model_config, path1=path1, path2=path2).to(device)
if "Combine" in model_config["model_name"]:
    model.m1 = model.m1.to(device)
    model.m2 = model.m2.to(device)

#model.load_state_dict(torch.load(f'models/BERT/model2.pt'))

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
        if data_config["data_type"] == "Word":
            x, m, t, y = datas
            x = x.to(device)
            m = m.to(device)
            t = t.to(device)
            y = y.to(device)
            pred = model(x, m, t)
        elif data_config["data_type"] == "Combine":
            xw, m, tt, xp, y = datas
            xw = xw.to(device)
            xp = xp.to(device)
            tt = tt.to(device)
            m = m.to(device)
            y = y.to(device)
            pred = model(xw, m, tt, xp)
        elif "Picture" in data_config["data_type"]:
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
            if data_config["data_type"] == "Word":
                x, m, t, y = datas
                x = x.to(device)
                m = m.to(device)
                t = t.to(device)
                y = y.to(device)
                pred = model(x, m, t)
            elif data_config["data_type"] == "Combine":
                xw, m, tt, xp, y = datas
                xw = xw.to(device)
                xp = xp.to(device)
                m = m.to(device)
                tt = tt.to(device)
                y = y.to(device)
                pred = model(xw, m, tt, xp)
            elif "Picture" in data_config["data_type"]:
                x, y = datas
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
            ans = torch.max(pred,1).indices
            predl.extend(pred.cpu().numpy())
            preds.extend(ans.cpu().numpy())
            true.extend(y.cpu().numpy())

    print(f'val_loss:{loss_fct(pred,y):.4f}')
    print(f'accuracy5:{myaccn(predl,true, 5)}')
    print(f'accuracy:{accuracy_score(preds,true)}')
    if save:
        torch.save(model.state_dict(), f'/home/F74106165/Language_Go/tmpmodels1/model{epoch+1}.pt')
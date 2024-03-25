import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
from myDatasets import get_datasets
from myModels import get_model



def myaccn(pred, true, n):
    total = len(true)
    correct = 0
    for i, p in tqdm(enumerate(pred), total=len(pred), leave=False):
        sorted_indices = (-p).argsort()
        top_k_indices = sorted_indices[:n]  
        if true[i] in top_k_indices:
            correct += 1
    return correct / total


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
    def forward(self, logits, target):
        log_probs = nn.functional.log_softmax(logits, dim=1)
        custom_loss = self.custom_nll_loss(log_probs, target)
        return custom_loss
    def custom_nll_loss(self, log_probs, target):
        _, indices = torch.sort(log_probs, descending=True, dim=1)
        rankings = torch.argsort(indices, dim=1)
        reward_weights = torch.ones_like(rankings, dtype=torch.float32)
        reward_weights[rankings == 9] = 1.2
        reward_weights[rankings == 8] = 1.2
        reward_weights[rankings == 7] = 1.2
        reward_weights[rankings == 6] = 1.2
        reward_weights[rankings == 5] = 1.2
        reward_weights[rankings == 4] = 1.5
        reward_weights[rankings == 3] = 1.5
        reward_weights[rankings == 2] = 2.0
        reward_weights[rankings == 1] = 2.0
        reward_weights[rankings == 0] = 3.0
        custom_loss = nn.functional.nll_loss(log_probs, target, reduction='none') * reward_weights[:, target]
        return custom_loss.mean()

data_config = {}
data_config["path"] = 'datas/data_240119.csv'
data_config["data_size"] = 300
data_config["offset"] = 0
data_config["data_type"] = "Word"
data_config["data_source"] = "pros"
data_config["num_moves"] = 80

model_config = {}
model_config["model_name"] = "BERT"
model_config["model_size"] = "mid"
model_config["config_path"] = "models_160/p1/config.json"
model_config["state_path"] = "models_160/p1/model.safetensors"

batch_size = 64
num_epochs = 50
lr = 5e-5
device = "cuda:1"
save = True
random_seed = 42


random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
model = get_model(model_config).to(device)

trainData, testData = get_datasets(data_config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fct = nn.CrossEntropyLoss()
#loss_fct = CustomCrossEntropyLoss()


train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testData, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    losses = []
    for datas in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        if data_config["data_type"] == "Word" or "posTest":
            x, m, y = datas
            x = x.to(device)
            m = m.to(device)
            y = y.to(device)
            pred = model(x, m)
        elif data_config["data_type"] == "Picture":
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
                x, m, y = datas
                x = x.to(device)
                m = m.to(device)
                y = y.to(device)
                pred = model(x, m)
            elif data_config["data_type"] == "Picture":
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
    print(f'accuracy10:{myaccn(predl,true,10)}')
    print(f'accuracy5:{myaccn(predl,true, 5)}')
    print(f'accuracy:{accuracy_score(preds,true)}')
    if save:
        torch.save(model.state_dict(), f'/home/F74106165/Transformer_Go/tmpmodels/model{epoch+1}.pt')
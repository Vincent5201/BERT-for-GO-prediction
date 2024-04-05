import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from get_datasets import get_datasets
from get_models import get_model

data_config = {}
data_config["path"] = 'datas/data_240119.csv'
data_config["data_size"] = 15000
data_config["offset"] = 15000
data_config["data_type"] = "Pretrain"
data_config["data_source"] = "pros"
data_config["num_moves"] = 240

model_config = {}
model_config["model_name"] = "pretrainxBERT"
model_config["model_size"] = "mid"

save_directory = '/home/F74106165/Transformer_Go/models_80/p3'
batch_size = 64
num_epochs = 50
lr = 5e-5
device = "cuda:1"
save = True

trainData, testData = get_datasets(data_config)
model = get_model(model_config)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
print("start")
for epoch in range(num_epochs):
    model.train()
    losses = []
    for datas in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        x, m, t, n, y = datas
        x = x.to(device)
        m = m.to(device)
        t = t.to(device)
        n = n.to(device)
        y = y.to(device)
        pred = model(input_ids=x, attention_mask=m, token_type_ids=t, next_sentence_label=n, labels=y)
        loss = pred.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'epoch {epoch+1}/{num_epochs},  train_loss = {sum(losses)/len(losses):.4f}')
    model.save_pretrained(save_directory)
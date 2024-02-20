import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from myDatasets import get_datasets
from myModels import get_model

batch_size = 64
num_epochs = 50
max_len = 80
lr = 5e-5
data_size = 140000
path = 'datas/data_Foxwq_9d.csv'
data_type = "Pretrain" 
data_source = "foxwq" 
num_moves = 80 
split_rate = 0.1
be_top_left = False
model_name = "BERT_pre"
model_size = "mid"
device = "cuda:1"

trainData, testData = get_datasets(path, data_type, data_source, data_size, num_moves, split_rate, be_top_left)
model = get_model(model_name, model_size)
model = model.to(device)
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
    torch.save(model.state_dict(), f'/home/F74106165/Transformer_Go/tmpmodels/model{epoch+1}.pt')
from transformers import BertModel, BertConfig, BertForPreTraining
import torch.nn as nn
import torch
import yaml
import torch.nn.functional as F

class Bert(nn.Module):
    def __init__(self, config, num_labels=361, p_model = None):
        super(Bert, self).__init__()
        if p_model:
            self.bert = p_model
        else:
            self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, 512)
        self.linear2 = nn.Linear(512, num_labels)
    def forward(self, x, m):
        output = self.bert(input_ids=x, attention_mask=m)["last_hidden_state"]
        logits = torch.mean(output, dim=1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits

class Bert_extend(nn.Module):
    def __init__(self, config, num_labels=361, p_model = None):
        super(Bert_extend, self).__init__()
        if p_model:
            self.bert = p_model
        else:
            self.bert = BertModel(config)
        self.linear1 = nn.Linear(config.hidden_size, 512)
        self.linear2 = nn.Linear(512, num_labels)
    def forward(self, x, m, t):
        output = self.bert(input_ids=x, attention_mask=m, token_type_ids=t)["last_hidden_state"]
        logits = torch.mean(output, dim=1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, padding=int((kernal_size-1)/2))
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  
        nn.init.kaiming_normal_(self.cnn.weight, mode="fan_out", nonlinearity="relu")
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x += self.beta.view(1, self.bn.num_features, 1, 1).expand_as(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.cnn1 = ConvBlock(in_channels, out_channels, 3)
        self.cnn2 = ConvBlock(out_channels, out_channels, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = F.relu(self.cnn1(x), inplace=True)
        out = self.cnn2(out)
        out += identity
        return F.relu(out, inplace=True)

class myResNet(nn.Module):
    def __init__(self, in_channels, res_channels, res_layers):
        super(myResNet, self).__init__()
        self.cnn_input = ConvBlock(in_channels, res_channels, 3)
        self.residual_tower = nn.Sequential(
            *[ResBlock(res_channels, res_channels) for _ in range(res_layers)]
        )
        self.policy_cnn = ConvBlock(res_channels, 2, 1)
        self.policy_fc = nn.Linear(2 * 19 * 19, 19 * 19)
    def forward(self, planes):
        x = self.cnn_input(planes)
        x = self.residual_tower(x)
        pol = self.policy_cnn(x)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))
        return pol

class Combine(nn.Module):
    def __init__(self, modelB, modelR):
        super(Combine, self).__init__()
        self.m1 = modelB
        self.m2 = modelR
        for param in self.m1.parameters():
            param.requires_grad = False
        for param in self.m2.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(722, 512)
        self.linear2 = nn.Linear(512, 361)
    def forward(self, xw, m, xp):
        yp = self.m2(xp)
        yw = self.m1(xw, m)
        yw = nn.functional.softmax(yw, dim=-1)
        yp = nn.functional.softmax(yp, dim=-1)
        logits = torch.cat((yp, yw), dim=-1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits

class CombineR(nn.Module):
    def __init__(self, modelR1, modelR2):
        super(CombineR, self).__init__()
        self.m1 = modelR1
        self.m2 = modelR2
        for param in self.m1.parameters():
            param.requires_grad = False
        for param in self.m2.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(722, 512)
        self.linear2 = nn.Linear(512, 361)
    def forward(self, x):
        y1 = self.m1(x)
        y2 = self.m2(x)
        y1 = nn.functional.softmax(y1, dim=-1)
        y2 = nn.functional.softmax(y2, dim=-1)
        logits = torch.cat((y1, y2), dim=-1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits

def get_model(model_config, path1=None, path2=None):
    with open('modelArgs.yaml', 'r') as file:
        args = yaml.safe_load(file)   

    if model_config["model_name"] == 'BERT':
        args = args["BERT"][model_config["model_size"]]
        config = BertConfig() 
        config.hidden_size = args["hidden_size"]
        config.num_hidden_layers = args["num_hidden_layers"]
        config.vocab_size = 364
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size*4
        config.position_embedding_type = "relative_key"
        model = Bert(config, 361)
    elif model_config["model_name"] == 'BERT_extend':
        args = args["BERT"][model_config["model_size"]]
        config = BertConfig()
        config.type_vocab_size = 7
        config.hidden_size = args["hidden_size"]
        config.num_hidden_layers = args["num_hidden_layers"]
        config.vocab_size = 363
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size*4
        config.position_embedding_type = "relative_key"
        model = Bert_extend(config, 361)
    elif model_config["model_name"] == 'ResNet':
        args = args["ResNet"][model_config["model_size"]]
        res_channel = args["res_channel"]
        layers = args["layers"]
        in_channel = 16
        model = myResNet(in_channel, res_channel, layers)
    elif model_config["model_name"] == 'CombineR':
        model_config["model_size"] = "mid"
        model_config["model_name"] = "ResNet"
        model1 = get_model(model_config)
        model1.load_state_dict(torch.load(path1))
        model2 = get_model(model_config)
        model2.load_state_dict(torch.load(path2))
        model = CombineR(model1, model2)
    elif model_config["model_name"] == 'Combine':
        model_config["model_size"] = "mid"
        model_config["model_name"] = "ResNet"
        model1 = get_model(model_config)
        model1.load_state_dict(torch.load(path1))
        model_config["model_name"] = "BERT"
        model2 = get_model(model_config)
        model2.load_state_dict(torch.load(path2))
        model = Combine(modelR=model1, modelB=model2)
    return model


if __name__ == "__main__":
    model_config = {}
    model_config["model_name"] = "BERT"
    model_config["model_size"] = "mid"
    model = get_model(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    print(model)

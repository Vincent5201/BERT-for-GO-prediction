from transformers import BertModel, BertConfig, BertForPreTraining
import torch.nn as nn
import torch
import yaml
import torch.nn.functional as F
from safetensors import safe_open

class Bert_Go(nn.Module):
    def __init__(self, config, num_labels, p_model = None):
        super(Bert_Go, self).__init__()
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

class BertCNN_Go(nn.Module):
    def __init__(self, config, num_labels, p_model = None):
        super(BertCNN_Go, self).__init__()
        if p_model:
            self.bert = p_model
        else:
            self.bert = BertModel(config)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(fs, config.hidden_size)) 
            for fs in range(2, 8)
        ])
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.linear1 = nn.Linear(6 * 32, 512)
        self.linear2 = nn.Linear(512, num_labels)
    
    def forward(self, x, m):
        outputs = self.bert(input_ids=x, attention_mask=m, output_hidden_states=True)["hidden_states"]
        concat = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in outputs[-4:]]), 0), 0, 1)
        convs = [F.relu(conv(concat)).squeeze(3) for conv in self.convs]
        pooleds = [self.pool(conv) for conv in convs]
        pooled_concat = torch.cat(pooleds, dim=2)       
        logits = pooled_concat.view(pooled_concat.size(0), -1)
        logits = self.linear1(logits)
        logits = self.linear2(logits)
        return logits


def get_model(model_config):
    with open('modelArgs.yaml', 'r') as file:
        args = yaml.safe_load(file)
    if not ("x" in model_config["model_name"]):
        args = args[model_config["model_name"]][model_config["model_size"]]

    if model_config["model_name"] == 'BERT':
        config = BertConfig() 
        config.hidden_size = args["hidden_size"]
        config.num_hidden_layers = args["num_hidden_layers"]
        config.vocab_size = 363
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size*4
        config.position_embedding_type = "relative_key"
        model = Bert_Go(config, 361)
    elif model_config["model_name"] == "BERTp":
        config = BertConfig() 
        config.hidden_size = args["hidden_size"]
        config.num_hidden_layers = args["num_hidden_layers"]
        config.vocab_size = 364
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size*4
        config.position_embedding_type = "relative_key"
        model = Bert_Go(config, 361)
    elif model_config["model_name"] == "BERTxpretrained":
        tensors = {}
        with safe_open(model_config["state_path"], framework="pt") as f:
            for k in f.keys():
                split_k = k.split('.')
                if split_k[0] == 'bert':
                    kk = k[5:]
                else:
                    kk = k
                tensors[kk] = f.get_tensor(k)
        keys_to_delete = ["cls.predictions.bias", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.transform.LayerNorm.weight",
                            "cls.predictions.transform.dense.bias", "cls.predictions.transform.dense.weight", "cls.seq_relationship.bias", "cls.seq_relationship.weight"]
        for key in keys_to_delete:
            del tensors[key]
        config = BertConfig.from_json_file(model_config["config_path"])
        pretrained_model = BertModel(config)
        pretrained_model.load_state_dict(tensors)
        model = Bert_Go(config, 361, pretrained_model)
    elif model_config["model_name"] == 'pretrainxBERT':
        args = args["BERT"][model_config["model_size"]]
        config = BertConfig() 
        config.hidden_size = args["hidden_size"]
        config.num_hidden_layers = args["num_hidden_layers"]
        config.vocab_size = 364
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size*4
        config.position_embedding_type = "relative_key"
        model = BertForPreTraining(config)
    elif model_config["model_name"] == 'ResNet':
        res_channel = args["res_channel"]
        layers = args["layers"]
        in_channel = 16
        model = myResNet(in_channel, res_channel, layers)
    elif model_config["model_name"] == 'BERTCNN':
        config = BertConfig() 
        config.hidden_size = args["hidden_size"]
        config.num_hidden_layers = args["num_hidden_layers"]
        config.vocab_size = 363
        config.num_attention_heads = 1
        config.intermediate_size = config.hidden_size*4
        config.position_embedding_type = "relative_key"
        model = BertCNN_Go(config, 361)
    
    return model

if __name__ == "__main__":
    model_config = {}
    model_config["model_name"] = "BERTCNN"
    model_config["model_size"] = "mid"
    model = get_model(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

from transformers import BertModel, DistilBertModel, RobertaModel
import torch
from torch import nn
import torch.nn.functional as F
import config


class GTMiner(nn.Module):
    def __init__(self,
                 device='cpu',
                 finetuning=True,
                 lm='bert',
                 n_relationships=2):
        super().__init__()

        if lm == 'bert':
            self.language_model = BertModel.from_pretrained(config.model_names[lm])
        elif lm == 'roberta':
            self.language_model = RobertaModel.from_pretrained(config.model_names[lm])
        elif lm == 'distilbert':
            self.language_model = DistilBertModel.from_pretrained(config.model_names[lm])
        else:
            lm = config.default_model
            self.language_model = BertModel.from_pretrained(config.model_names[lm])

        self.device = device
        self.finetuning = finetuning
        self.n_relationships = n_relationships

        self.drop = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(config.hidden_size + 2*config.co_emb, (config.hidden_size + 2*config.co_emb)//2)
        self.linear2 = nn.Linear((config.hidden_size + 2*config.co_emb)//2, self.n_relationships)
        self.co_linear = nn.Linear(1, 2*config.co_emb)
        self.w_att = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_linear = nn.Linear(config.hidden_size + 2*config.co_emb, 1)

        self.relu = nn.ReLU()

    def forward(self, x, d, att_mask, training=True):

        x = x.to(self.device)
        d = d.to(self.device)
        att_mask = att_mask.to(self.device)

        if len(x.shape) < 2:
            x = x.unsqueeze(0)
            att_mask = att_mask.unsqueeze(0)

        if training and self.finetuning:
            self.language_model.train()
            self.train()
            output = self.language_model(x, attention_mask=att_mask)
            pooled_output = output[0][:, :, :]

        else:
            self.language_model.eval()
            with torch.no_grad():
                output = self.language_model(x, attention_mask=att_mask)
                pooled_output = output[0][:, :, :]

        d = d.unsqueeze(1)
        d = self.co_linear(d)
        cword_d = torch.cat([self.w_att(pooled_output), d.unsqueeze(1).repeat(1, pooled_output.shape[1], 1)], -1)
        att = F.softmax(self.att_linear(cword_d), -2)
        cword_att = cword_d*att
        cword_agg = torch.sum(cword_att, 1)
        output = self.linear2(self.drop(self.relu(self.linear1(cword_agg))))

        return F.log_softmax(output, dim=1)


class BertFE(nn.Module):
    def __init__(self,
                 device='cpu',
                 finetuning=True,
                 lm='bert'):
        super().__init__()

        self.language_model = BertModel.from_pretrained(config.model_names[lm])
        self.device = device
        self.finetuning = finetuning

        self.drop = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.linear2 = nn.Linear(config.hidden_size // 2, 2)

        self.relu = nn.ReLU()

    def forward(self, x, att_mask, training=True):

        x = x.to(self.device)
        att_mask = att_mask.to(self.device)

        if len(x.shape) < 2:
            x = x.unsqueeze(0)
            att_mask = att_mask.unsqueeze(0)

        if training and self.finetuning:
            self.language_model.train()
            self.train()
            output = self.language_model(x, attention_mask=att_mask)
            pooled_output = output[0][:, 0, :]

        else:

            self.language_model.eval()
            with torch.no_grad():
                output = self.language_model(x, attention_mask=att_mask)
                pooled_output = output[0][:, 0, :]

        output = self.linear2(self.drop(self.relu(self.linear1(pooled_output))))

        return F.log_softmax(output, dim=1)


class LSTMFE(nn.Module):
    def __init__(self,
                 device='cpu',
                 input_size=100,
                 hidden_size=64,
                 bidirectional=True,
                 max_seq_len=16,
                 dropout=0.1):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size

        if bidirectional:
            self.b = 2
        else:
            self.b = 1

        self.drop = nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.linear1 = nn.Linear(self.b*self.hidden_size * max_seq_len, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.to(self.device)

        output, (_, _) = self.lstm(x)

        output = self.linear2(self.drop(self.relu(self.linear1(output.reshape(output.shape[0], -1)))))

        return F.log_softmax(output, dim=1)

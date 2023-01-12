from io import open
from io import open
import torch
from torch import nn, optim
from sklearn.metrics import f1_score
import numpy as np
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
import random
import time
import pickle
import config
from functions import *
from refinement import extend, repair


def prepare_dataset(path, max_seq_len=128, tokenizer='bert'):

    if tokenizer == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(config.model_names[tokenizer])
    elif tokenizer == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(config.model_names[tokenizer])
    else:
        tokenizer = BertTokenizer.from_pretrained(config.model_names[config.default_model])

    data_x = []
    data_y = []
    data_d = []
    data_ids = []

    df = pd.read_csv(path, index_col=0)
    cols = df.columns

    for i in range(df.shape[0]):

        this_row = df.iloc[i]

        id1, lat1, lon1, e1, id2, lat2, lon2, e2 = textualize(this_row, cols)

        d = compute_dist(lat1, lon1, lat2, lon2)
        try:
            d = int(d)
        except ValueError:
            d = config.error_d

        d = norm_d(d)

        x = tokenizer.tokenize('[CLS] ' + e1 + ' [SEP] ' + e2 + ' [SEP]')

        y = int(this_row[-1])

        if len(x) < max_seq_len:
            x = x + ['[PAD]'] * (max_seq_len - len(x))
        else:
            x = x[:max_seq_len]

        data_x.append(tokenizer.convert_tokens_to_ids(x))
        data_y.append(y)
        data_d.append(d)
        data_ids.append([id1, id2])

    return [torch.tensor(data_x), torch.tensor(data_d), torch.tensor(data_y), data_ids]


def prepare_dataset_BertFE(path, max_seq_len=32):

    tokenizer = BertTokenizer.from_pretrained(config.model_names['bert'])
    data_x = []
    data_y = []

    df = pd.read_csv(path, index_col=0)
    cols = df.columns

    for i in range(df.shape[0]):

        this_row = df.iloc[i]

        e1 = textualize_block(this_row, cols)

        x = tokenizer.tokenize('[CLS] ' + e1 + ' [SEP]')

        y = int(this_row[-1])

        if len(x) < max_seq_len:
            x = x + ['[PAD]'] * (max_seq_len - len(x))
        else:
            x = x[:max_seq_len]

        data_x.append(tokenizer.convert_tokens_to_ids(x))
        data_y.append(y)

    return torch.tensor(data_x), torch.tensor(data_y)


def prepare_dataset_LSTMFE(path, glove_model, max_seq_len=16):
    data_x = []
    data_y = []

    unk_vector = np.mean(list(glove_model.values()), axis=0)

    df = pd.read_csv(path, index_col=0)
    cols = df.columns

    for i in range(df.shape[0]):

        this_row = df.iloc[i]

        e1 = tokenize_block(this_row, cols)

        y = int(this_row[-1])

        if len(e1) < max_seq_len:
            e1 = e1 + ['padding'] * (max_seq_len - len(e1))
        else:
            e1 = e1[:max_seq_len]

        x1 = []

        for word in e1:
            if word in glove_model.keys():
                x1.append(glove_model[word])
            else:
                x1.append(unk_vector)

        data_x.append(x1)
        data_y.append(y)

    return torch.tensor(np.array(data_x)).double(), torch.tensor(data_y)


def train(model, train_data, valid_data, test_data, save_path, hp):
    train_x_tensor, train_d_tensor, train_y_tensor, _ = [train_data[i] for i in range(len(train_data))]
    attention_mask = np.where(train_x_tensor != 0, 1, 0)
    attention_mask = torch.tensor(attention_mask)

    opt = optim.Adam(params=model.parameters(), lr=hp.lr)
    criterion = nn.NLLLoss()
    best_f1 = 0
    b_s = hp.batch_size

    for epoch in range(hp.n_epochs):

        model.train()

        print('\n==========  EPOCH:', epoch + 1, ' ==========\n')
        i = 0
        step = 1

        while i < train_x_tensor.shape[0]:

            model.zero_grad()

            if i + b_s > train_x_tensor.shape[0]:
                y = train_y_tensor[i:].view(-1).to(hp.device)
                x = train_x_tensor[i:]
                d = train_d_tensor[i:]
                att_mask = attention_mask[i:]
            else:
                y = train_y_tensor[i: i + b_s].view(-1).to(hp.device)
                x = train_x_tensor[i: i + b_s]
                d = train_d_tensor[i: i + b_s]
                att_mask = attention_mask[i: i + b_s]

            pred = model(x, d, att_mask)

            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            if hp.device == 'cuda':
                print('Step:', step, 'Loss:', loss.cpu().detach().numpy())
            else:
                print('Step:', step, 'Loss:', loss.item())

            step += 1
            i += b_s

        print('\n==========  Validation Epoch:', epoch + 1, ' ==========\n')
        valid_f1 = validate(model, valid_data, b_s, hp.device, False, False)

        print('\n==========  Test Epoch:', epoch + 1, ' ==========\n')
        _ = validate(model, test_data, b_s, hp.device, hp.do_extend, hp.do_repair)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            if hp.save_model:
                pickle.dump(model, open(save_path, 'wb'))


def train_BertFE(model, train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, test_x_tensor, test_y_tensor,
                 save_path, hp):

    attention_mask = np.where(train_x_tensor != 0, 1, 0)
    attention_mask = torch.tensor(attention_mask)

    opt = optim.Adam(params=model.parameters(), lr=hp.lr)
    criterion = nn.NLLLoss(weight=torch.tensor([hp.beta, hp.alpha]).to(hp.device))

    best_f1 = 0
    b_s = hp.batch_size

    for epoch in range(hp.n_epochs):

        model.train()

        print('\n==========  EPOCH:', epoch + 1, ' ==========\n')
        i = 0
        step = 1

        while i < train_x_tensor.shape[0]:

            model.zero_grad()

            if i + b_s > train_x_tensor.shape[0]:
                y = train_y_tensor[i:].view(-1).to(hp.device)
                x = train_x_tensor[i:]
                att_mask = attention_mask[i:]
            else:
                y = train_y_tensor[i: i + b_s].view(-1).to(hp.device)
                x = train_x_tensor[i: i + b_s]
                att_mask = attention_mask[i: i + b_s]

            pred = model(x, att_mask)

            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            if hp.device == 'cuda':
                print('Step:', step, 'Loss:', loss.cpu().detach().numpy())
            else:
                print('Step:', step, 'Loss:', loss.item())

            step += 1
            i += b_s

        print('\n==========  Validation Epoch:', epoch + 1, ' ==========\n')
        valid_f1 = validate_BertFE(model, valid_x_tensor, valid_y_tensor, b_s, hp.device)

        print('\n==========  Test Epoch:', epoch + 1, ' ==========\n')
        _ = validate_BertFE(model, test_x_tensor, test_y_tensor, b_s, hp.device)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            pickle.dump(model, open(save_path, 'wb'))


def train_LSTMFE(model, train_x_tensor, train_y_tensor, valid_x_tensor, valid_y_tensor, test_x_tensor, test_y_tensor,
                 save_path, hp):

    opt = optim.Adam(params=model.parameters(), lr=hp.lr)
    criterion = nn.NLLLoss(weight=torch.tensor([hp.beta, hp.alpha]).double().to(hp.device))

    best_f1 = 0
    b_s = hp.batch_size

    for epoch in range(hp.n_epochs):

        model.train()

        print('\n==========  EPOCH:', epoch + 1, ' ==========\n')
        i = 0
        step = 1

        while i < train_x_tensor.shape[0]:

            model.zero_grad()

            if i + b_s > train_x_tensor.shape[0]:
                y = train_y_tensor[i:].view(-1).to(hp.device)
                x = train_x_tensor[i:]
            else:
                y = train_y_tensor[i: i + b_s].view(-1).to(hp.device)
                x = train_x_tensor[i: i + b_s]

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            if hp.device == 'cuda':
                print('Step:', step, 'Loss:', loss.cpu().detach().numpy())
            else:
                print('Step:', step, 'Loss:', loss.item())

            step += 1
            i += b_s

        print('\n==========  Validation Epoch:', epoch + 1, ' ==========\n')
        valid_f1 = validate_LSTMFE(model, valid_x_tensor, valid_y_tensor, b_s, hp.device)

        print('\n==========  Test Epoch:', epoch + 1, ' ==========\n')
        _ = validate_LSTMFE(model, test_x_tensor, test_y_tensor, b_s, hp.device)

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            pickle.dump(model, open(save_path, 'wb'))


def validate(model, valid_data, b_s, device, do_extend, do_repair):

    valid_x_tensor, valid_d_tensor, valid_y_tensor, valid_ids = [valid_data[i] for i in range(len(valid_data))]
    attention_mask = np.where(valid_x_tensor != 0, 1, 0)
    attention_mask = torch.tensor(attention_mask)

    model.eval()
    y_pred = np.array([])
    y_probs = np.array([])
    i = 0
    while i < valid_x_tensor.shape[0]:

        if i + b_s > valid_x_tensor.shape[0]:
            x = valid_x_tensor[i:]
            d = valid_d_tensor[i:]
            att_mask = attention_mask[i:]
        else:
            x = valid_x_tensor[i: i + b_s]
            d = valid_d_tensor[i: i + b_s]
            att_mask = attention_mask[i: i + b_s]

        y_p = model(x, d, att_mask, training=False)
        if device == 'cuda':
            y_prob = np.exp(np.max(y_p.detach().cpu().numpy(), 1))
            y_p = torch.argmax(y_p, 1).detach().cpu().numpy()
        else:
            y_prob = np.exp(np.max(y_p.detach().numpy(), 1))
            y_p = torch.argmax(y_p, 1).detach().numpy()
        y_pred = np.concatenate([y_pred, y_p])
        y_probs = np.concatenate([y_probs, y_prob])

        i += b_s

    if do_repair or do_extend:
        print('Refinement...')

    if do_repair:
        print('Performing Repair...')
        repair(y_pred, y_probs, valid_ids)

    if do_extend:
        print('Performing Extend...')
        extend(y_pred, valid_ids)

    tot_p = 0
    true_p = 0
    pred_p = 0
    for i in range(len(y_pred)):

        if valid_y_tensor[i] > 0:
            tot_p += 1

            if y_pred[i] == valid_y_tensor[i]:
                true_p += 1

        if y_pred[i] > 0:
            pred_p += 1

    f1 = 0.0
    prec = 0.0
    rec = 0.0

    if tot_p and pred_p:
        rec = true_p / tot_p
        prec = true_p / pred_p

        if rec > 0 or prec > 0:
            f1 = 2 * prec * rec / (prec + rec)

    print('P: ' + str(round(prec, 4)) + '  |  R: ' + str(round(rec, 4)) + '  |  F1: ' + str(round(f1, 4)))

    return f1


def validate_BertFE(model, valid_x_tensor, valid_y_tensor, b_s, device):
    attention_mask = np.where(valid_x_tensor != 0, 1, 0)
    attention_mask = torch.tensor(attention_mask)

    model.eval()
    y_pred = np.array([])
    i = 0
    while i < valid_x_tensor.shape[0]:

        if i + b_s > valid_x_tensor.shape[0]:
            x = valid_x_tensor[i:]
            att_mask = attention_mask[i:]
        else:
            x = valid_x_tensor[i: i + b_s]
            att_mask = attention_mask[i: i + b_s]

        y_p = model(x, att_mask, training=False)
        if device == 'cuda':
            y_p = torch.argmax(y_p, 1).detach().cpu().numpy()
        else:
            y_p = torch.argmax(y_p, 1).detach().numpy()
        y_pred = np.concatenate([y_pred, y_p])

        i += b_s

    tot_p = 0
    true_p = 0
    pred_p = 0
    for i in range(len(y_pred)):

        if valid_y_tensor[i] > 0:
            tot_p += 1

            if y_pred[i] == valid_y_tensor[i]:
                true_p += 1

        if y_pred[i] > 0:
            pred_p += 1

    f1 = 0.0
    prec = 0.0
    rec = 0.0

    if tot_p and pred_p:
        rec = true_p / tot_p
        prec = true_p / pred_p

        if rec > 0 or prec > 0:
            f1 = 2 * prec * rec / (prec + rec)

    print('P: ' + str(round(prec, 4)) + '  |  R: ' + str(round(rec, 4)) + '  |  F1: ' + str(round(f1, 4)))

    return f1


def validate_LSTMFE(model, valid_x_tensor, valid_y_tensor, b_s, device):
    model.eval()
    y_pred = np.array([])
    i = 0
    while i < valid_x_tensor.shape[0]:

        if i + b_s > valid_x_tensor.shape[0]:
            x = valid_x_tensor[i:]
        else:
            x = valid_x_tensor[i: i + b_s]

        y_p = model(x)
        if device == 'cuda':
            y_p = torch.argmax(y_p, 1).detach().cpu().numpy()
        else:
            y_p = torch.argmax(y_p, 1).detach().numpy()
        y_pred = np.concatenate([y_pred, y_p])

        i += b_s

    tot_p = 0
    true_p = 0
    pred_p = 0

    for i in range(len(y_pred)):

        if valid_y_tensor[i] > 0:
            tot_p += 1

            if y_pred[i] == valid_y_tensor[i]:
                true_p += 1

        if y_pred[i] > 0:
            pred_p += 1

    f1 = 0.0
    prec = 0.0
    rec = 0.0

    if tot_p and pred_p:
        rec = true_p / tot_p
        prec = true_p / pred_p

        if rec > 0 or prec > 0:
            f1 = 2 * prec * rec / (prec + rec)

    print('P: ' + str(round(prec, 4)) + '  |  R: ' + str(round(rec, 4)) + '  |  F1: ' + str(round(f1, 4)))

    return f1

import random
import numpy as np
import torch
from param import *
from utilss import get_link_labels

args = parameter_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    # 训练集负样本
    train_neg_edge_index = np.mat(np.where(data.train_matrix.cpu().numpy() < 1)).T.tolist()
    random.shuffle(train_neg_edge_index)
    train_neg_edge_index = train_neg_edge_index[:data.train_pos_edge_index.shape[1]]
    train_neg_edge_index = np.array(train_neg_edge_index).T
    train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)  # tensor格式训练集负样本

    output = model(data.x, data.Adj, data.edge_index, data.edge_attr)
    edge_index = torch.cat([data.train_pos_edge_index, train_neg_edge_index], 1)#边索引
    trian_scores = output[edge_index[0], edge_index[1]].to(device)
    trian_labels = get_link_labels(data.train_pos_edge_index, train_neg_edge_index).to(device)  # 训练集中正样本标签
    loss = criterion(trian_scores, trian_labels)
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss, train_neg_edge_index, trian_labels, trian_scores, output
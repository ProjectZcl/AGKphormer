import csv
import os

import torch as torch
from train import *
from test import *
from param import *
from utilss import *
import numpy as np
import pandas as pd
import random
import warnings

args = parameter_parser()  # 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_load(k):
    Adj = pd.read_csv('../data/association_matrix.csv', header=0) #2315*265
    Dis_simi = pd.read_csv('../data/diease_network_simi.csv', header=0) #265*265
    Meta_simi = pd.read_csv('../data/metabolite_ntework_simi.csv', header=0) #2315*2315
    # Adj_next = pd.read_csv('../data/association_matrix.csv', header=0)
    Adj_next = pd.read_csv('../data/association_matrix_completed_admm.csv', header=0)


    one_matrix = np.mat(np.where(Adj_next == 1))  # 输出邻接矩阵中为“1”的关联关系，维度：2 X 4763
    association_num = one_matrix.shape[1]  # 关联关系数：4763
    random_one = one_matrix.T.tolist()  # list：4763 X 2
    random.seed(args.seed)  # random.seed(): 设定随机种子，使得random.shuffle随机打乱的顺序一致
    random.shuffle(random_one)  # random.shuffle将random_index列表中的元素打乱顺序
    k_folds = args.k_folds
    CV_size = int(association_num / k_folds)  # 每折的个数
    temp = np.array(random_one[:association_num - association_num % k_folds]).reshape(k_folds, CV_size, -1).tolist()  # %取余,每折分952个，结果存储在temp中
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_one[association_num - association_num % k_folds:]  # 将余下的元素加到最后一折里面
    random_index = temp
    metric = np.zeros((1, 7))

    train_matrix = np.matrix(Adj_next, copy=True)  # 将邻接矩阵转化为np矩阵
    train_matrix_root = np.matrix(Adj,copy=True)
    train_matrix_root[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0
    train_matrix[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0

    #节点特征矩阵与边索引、边权值
    x = constructHNet(train_matrix_root)
    adj = constructGNet(train_matrix, Meta_simi, Dis_simi)
    adj_list = adj.tolist()
    edge_index = np.array(np.where(adj > 0))#边索引
    edge_attr_list = []
    for i in range(len(edge_index[0])):
        row = edge_index[0][i]
        col = edge_index[1][i]
        edge_attr_list.append(adj_list[row][col])

    #测试集正负样本
    val_pos_edge_index = np.array(random_index[k]).T  # 验证集边索引，正样本
    # 验证集负采样，采集与正样本相同数量的负样本
    val_neg_edge_index = np.mat(np.where(train_matrix_root < 1)).T.tolist()
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = val_neg_edge_index[:val_pos_edge_index.shape[1]]
    val_neg_edge_index = np.array(val_neg_edge_index).T

    #训练集正样本，训练集负样本在epoch内划分，保证更好的可解释性
    train_pos_edge_index = np.mat(np.where(train_matrix_root > 0))  # 训练集边索引，正样本

    data = Data(
        Meta_simi = Meta_simi,
        Adj = torch.tensor(adj, dtype=torch.float).to(device),
        Adj_next = Adj_next,
        train_matrix = torch.tensor(train_matrix, dtype=torch.float).to(device),
        x = torch.tensor(x, dtype=torch.float).to(device),
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device),
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(device),  # 边权值
        val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).to(device),  # tensor格式，验证集负样本
        val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long).to(device),  # tensor格式，验证集正样本
        train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)  # tensor格式，训练集正样本
    )
    return data
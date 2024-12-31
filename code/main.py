import torch as torch
from mpmath import degree
from model import *
from train import *
from test import *
from param import *
from utilss import *
from data_load import *
import numpy as np
import pandas as pd
import random
import time

args = parameter_parser()  # 超参数
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
auc_result = []
acc_result = []
pre_result = []
recall_result = []
f1_result = []
prc_result = []
fprs = []
tprs = []
precisions = []
recalls = []

print("seed=%d, evaluating met-disease...." % args.seed)
for k in range(args.k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    data = data_load(k)

    model = GKformer(in_channels = data.x.shape[1],
                     hidden_channels = args.hidden_channels,
                     out_channels = args.out_channels,
                     heads = args.heads,
                     dropout_trans=args.dropout_trans,
                     num_gcn_layers = args.num_gcn_layers,
                     num_transformer_layers = args.num_transformer_layers,
                     dropout = args.dropout,
                     num_r = data.Meta_simi.shape[0]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion =  F.binary_cross_entropy
    best_auc = best_prc = best_epoch = best_tpr = best_fpr = best_recall = best_precision = 0

    for epoch in range(args.epochs):
        start_time = time.time()  # 记录epoch开始时间
        train_loss, train_neg_edge_index, trian_labels, trian_scores, output = train(data, model, optimizer, criterion)
        score_val, metric_tmp, train_auc, val_auc, val_prc, tpr, fpr, recall, precision = mytest(data, trian_labels,trian_scores, model,output, train_neg_edge_index)
        end_time = time.time()  # 记录epoch结束时间
        epoch_time = end_time - start_time  # 计算epoch所用时间
        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % train_loss.item(),
              'Acc: %.4f' % metric_tmp[0], 'Pre: %.4f' % metric_tmp[1], 'Recall: %.4f' % metric_tmp[2],
              'F1: %.4f' % metric_tmp[3],
              'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Val PRC: %.4f' % val_prc,
              'Time: %.2f' % (end_time - start_time))

        if val_auc > best_auc:
            metric_tmp_best = metric_tmp
            best_auc = val_auc
            best_prc = val_prc
            best_epoch = epoch + 1
            best_tpr = tpr
            best_fpr = fpr
            best_recall = recall
            best_precision = precision

    print('Fold:', k + 1, 'Best Epoch:', best_epoch, 'Val acc: %.4f' % metric_tmp_best[0],
          'Val Pre: %.4f' % metric_tmp_best[1],
          'Val Recall: %.4f' % metric_tmp_best[2], 'Val F1: %.4f' % metric_tmp_best[3], 'Val AUC: %.4f' % best_auc,
          'Val PRC: %.4f' % best_prc,
          )

    acc_result.append(metric_tmp_best[0])
    pre_result.append(metric_tmp_best[1])
    recall_result.append(metric_tmp_best[2])
    f1_result.append(metric_tmp_best[3])
    auc_result.append(best_auc)
    prc_result.append(best_prc)
    fprs.append(best_fpr)
    tprs.append(best_tpr)
    recalls.append(best_recall)
    precisions.append(best_precision)

print('## Training Finished !')
print('-----------------------------------------------------------------------------------------------')
print('Acc', acc_result)
print('Pre', pre_result)
print('Recall', recall_result)
print('F1', f1_result)
print('Auc', auc_result)
print('Prc', prc_result)
print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
        'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
        'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
        'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
        'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
        'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

pd.DataFrame(recalls).to_csv('../output/recalls.csv', index=False)
pd.DataFrame(precisions).to_csv('../output/precisions.csv', index=False)
pd.DataFrame(fprs).to_csv('../output/fprs.csv', index=False)
pd.DataFrame(tprs).to_csv('../output/tprs.csv', index=False)
print('fprs', fprs)
print('tprs', tprs)
# print('recalls', recalls)
# print('precisions', precisions)
# 画五折AUC和PR曲线
plot_auc_curves(fprs, tprs, auc_result, directory='../output', name='test_auc')
plot_prc_curves(precisions, recalls, prc_result, directory='../output', name='test_prc')

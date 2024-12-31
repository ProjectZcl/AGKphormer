import numpy as np
import torch
from sklearn import metrics
from utilss import *
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@torch.no_grad()
def mytest(data,trian_labels,trian_scores, model,output, train_neg_edge_index):
    model.eval()

    with torch.no_grad():  # 禁用梯度计算，以避免跟踪计算图中的梯度
        score_train_cpu = np.squeeze(trian_scores.detach().cpu().numpy())
        label_train_cpu = np.squeeze(trian_labels.detach().cpu().numpy())
        train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)

        predict_y_proba = output.reshape(data.Adj_next.shape[0], data.Adj_next.shape[1]).to(device)
        score_val, label_val, metric_tmp = cv_model_evaluate(predict_y_proba, data.val_pos_edge_index, data.val_neg_edge_index)

        fpr, tpr, thresholds = metrics.roc_curve(label_val, score_val)
        precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
        val_auc = metrics.auc(fpr, tpr)
        val_prc = metrics.auc(recall, precision)

        return score_val, metric_tmp, train_auc, val_auc, val_prc, tpr, fpr, recall, precision


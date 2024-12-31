import torch
import numpy as np
from torch_geometric.data import Data
import scipy.sparse as sp
import matplotlib.pyplot as plt
#from scipy import interp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def constructGNet(met_dis_matrix,met_matrix,dis_matrix):
    mat1 = np.hstack((met_matrix, met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(met_dis_matrix):
    met_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[0],met_dis_matrix.shape[0]),dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((met_dis_matrix.shape[1],met_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((met_matrix,met_dis_matrix))
    mat2 = np.hstack((met_dis_matrix.T,dis_matrix))
    adj = np.vstack((mat1,mat2))
    return adj

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(predict_score.flatten()))))  # set只保留唯一值，并从小到大排序
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]  # 抽取999个作为阈值
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))  # 将predict_score复制hresholds_num（999）次
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)  # 正确预测为正样本的数量（真阳率）
    FP = predict_score_matrix.sum(axis=1) - TP  # 错误预测为正样本的数量  求和表示所有正样本个数
    FN = real_score.sum() - TP  # 错误预测为负样本的数量
    TN = len(real_score.T) - TP - FP - FN  # 正确预测为负样本的数量（真阴率）

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [accuracy, precision, recall, f1_score]


#  interaction_matrix原邻接矩阵4536个1  predict_matrix预测邻接矩阵  train_matrix 去掉测试集的训练矩阵4536-907=3629个1
def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_pos_edge_index).to(device)  # 训练集中正样本标签
    return val_scores.cpu().numpy(), val_labels.cpu().numpy(), get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy())



def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i])) #原interp
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='#cb0000',  alpha=0.9, label='Mean AUC: %.4f' % mean_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)

    # std_tpr = np.std(tpr, axis=0)
    # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='LightSkyBlue', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))#原interp
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='#cb0000', alpha=0.9,
             label='Mean AUPR: %.4f' % mean_prc)  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.close()


import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Acc', 'Pre', 'Recall', 'F1', 'AUPR', 'AUC']
gcn_a = [87.78, 86.79, 89.19, 87.96, 94.00, 94.51]
gcn_a_sim = [90.48, 90.24, 90.83, 90.51, 96.15, 96.52]
wgcncdlc = [91.68, 90.91, 92.61, 91.75, 97.40, 97.43]

# 设置柱状图的位置
x = np.arange(len(categories))
width = 0.25

# 创建柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, gcn_a, width, label='GCN(A)')
rects2 = ax.bar(x, gcn_a_sim, width, label='GCN(A+Sim)')
rects3 = ax.bar(x + width, wgcncdlc, width, label='WGCNCDLC')

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 添加一些文本用于标签、标题和自定义x轴刻度等
ax.set_ylabel('%')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 显示图表
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator

# 数据
categories = ['Acc', 'Pre', 'Recall', 'F1']
AGKformer = [91.52, 90.75, 92.48, 91.60]
mdbirw = [80.13, 79.02, 82.90, 80.89]
dwrf = [88.01, 90.27, 85.34, 87.74]
Deep_DRM = [89.20, 87.59, 90.21, 89.69]
gcnat = [88.22, 87.70, 89.14, 88.34]
MDA_AENMF = [90.75, 89.95, 91.87, 90.86]



# 设置柱状图的位置
x = np.arange(len(categories))
width = 0.12

colors = ['#ADD8E6', '#90EE90', '#E6E6FA', '#FFC0CB', '#D3D3D3']

# 创建柱状图，调整figsize使图表变宽
fig, ax = plt.subplots(figsize=(10, 5.5))  # 将宽度从6调整为10
rects1 = ax.bar(x - 2*width, AGKformer, width, label='AGKformer', color='#FF7171')
rects2 = ax.bar(x - width, mdbirw, width, label='MDBIRW', color='#ADD8E6')
rects3 = ax.bar(x, dwrf, width, label='DWRF', color='#E6E6FA')
rects4 = ax.bar(x + width, Deep_DRM, width, label='Deep-DRM', color='#87CEEB')
rects5 = ax.bar(x + 2*width, gcnat, width, label='GCNAT', color='#FFC0CB')
rects6 = ax.bar(x + 3*width, MDA_AENMF, width, label='MDA-AENMF', color='#D3D3D3')

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

ax.yaxis.set_major_locator(MaxNLocator(5))  # Y轴最大刻度数量
ax.set_ylim(76, 97)

# 添加一些文本用于标签、标题和自定义x轴刻度等
ax.set_ylabel('%')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)

# 设置图例在上方，水平排列
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6)

plt.savefig('comparison_perf.pdf', format='pdf', bbox_inches='tight')
# 显示图表
plt.show()
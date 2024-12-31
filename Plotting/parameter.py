import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib.ticker import MultipleLocator

# GCN_layer数据
x = [1, 2, 3, 4, 5]
y1 = [97.27, 97.32, 96.59, 96.04, 95.92]
y2 = [97.07, 97.34, 96.23, 96.05, 95.70]

# 输出维度数据
# x = [16, 32, 64, 128, 256]
# y1 = [97.10, 97.05, 97.28, 97.32, 97.26]
# y2 = [97.07, 96.94, 97.27, 97.34, 97.12]

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 7.5))

# 绘制第一条折线图
ax.plot(x, y1, label='AUC', color='#076CFB', linestyle='-', linewidth=3, marker='o', markeredgewidth=4, markeredgecolor='white', markersize=16)
# 绘制第二条折线图
ax.plot(x, y2, label='AUPR', color='#FB6007', linestyle='-', linewidth=3, marker='s', markeredgewidth=4, markeredgecolor='white', markersize=16)

# 添加标题和轴标签
# ax.set_title('(a)', fontsize=16, fontweight='bold', color='#2c3e50', )
ax.set_xlabel('GCN layers', fontsize=20, color='black')
ax.set_ylabel('Average AUC / APR(%)', fontsize=20, color='black')

# 设置图例
ax.legend(loc='upper right', fontsize=20)

ax.yaxis.set_major_locator(FixedLocator([0, 95, 96, 97, 98]))
ax.xaxis.set_major_locator(MaxNLocator(5))  # X轴最大刻度数量
ax.yaxis.set_major_locator(MaxNLocator(5))  # Y轴最大刻度数量

# 设置网格线样式
ax.grid(True, linestyle='--', color='#bdc3c7', linewidth=0.8)

ax.set_xlim(0.7, 5.3)  # 稍微扩展范围以避免切割标记
ax.set_ylim(95.6, 97.4)  # 设置Y轴的范围

# ax.xaxis.set_major_locator(FixedLocator(x))
# ax.set_xscale('log')
# ax.set_xlim(16, )  # 稍微扩展范围以避免切割标记
# ax.set_ylim(96.9, 97.4)  # 设置Y轴的范围



# 设置轴刻度的样式
ax.tick_params(axis='x', colors='black', labelsize=18, tickdir='in', pad=10)
ax.tick_params(axis='y', colors='black', labelsize=18, tickdir='in', pad=10)


# 添加数据标签
# for i, txt in enumerate(y1):
#     ax.annotate(f"{txt:.2f}", (x[i], y1[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#7f8c8d', size=15)
# for i, txt in enumerate(y2):
#     ax.annotate(f"{txt:.2f}", (x[i], y2[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#7f8c8d', size=15)

# 保存图形
plt.savefig('GCN_layer.png', dpi=300, bbox_inches='tight')
plt.savefig('GCN_layer.pdf', dpi=300, bbox_inches='tight')

# plt.savefig('out_dim.png', dpi=300, bbox_inches='tight')
# plt.savefig('out_dim.pdf', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator

# 输出维度数据
x_values = [16, 32, 64, 128, 256]  # 原始x值
y1 = [97.10, 97.05, 97.28, 97.32, 97.26]
y2 = [97.07, 96.94, 97.27, 97.34, 97.12]

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 7.5))

# 使用linspace生成等间距的x坐标位置
x_positions = np.linspace(0, len(x_values) - 1, len(x_values))

# 绘制第一条折线图
ax.plot(x_positions, y1, label='AUC', color='#076CFB', linestyle='-', linewidth=3, marker='o', markeredgewidth=4, markeredgecolor='white', markersize=16)
# 绘制第二条折线图
ax.plot(x_positions, y2, label='AUPR', color='#FB6007', linestyle='-', linewidth=3, marker='s', markeredgewidth=4, markeredgecolor='white', markersize=16)

# 添加标题和轴标签
# ax.set_title('(b)', fontsize=16, fontweight='bold', color='#2c3e50')
ax.set_xlabel('output dimension', fontsize=20, color='black')
ax.set_ylabel('Average AUC / APR(%)', fontsize=20, color='black')

# 设置图例
ax.legend(loc='upper left', fontsize=20)
ax.yaxis.set_major_locator(MaxNLocator(6))  # Y轴最大刻度数量

# 设置x轴的刻度值和标签
ax.set_xticks(x_positions)
ax.set_xticklabels(x_values)

# 设置网格线样式
ax.grid(True, linestyle='--', color='#bdc3c7', linewidth=0.8)

# 设置轴刻度的样式
ax.tick_params(axis='x', colors='black', labelsize=18, tickdir='in', pad=10)
ax.tick_params(axis='y', colors='black', labelsize=18, tickdir='in', pad=10)

# 保存图形
plt.savefig('heads.png', dpi=300, bbox_inches='tight')
plt.savefig('heads.pdf', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
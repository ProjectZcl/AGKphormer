import matplotlib.pyplot as plt

# 数据
models = ['AGKformer', 'Del-ADMM', 'Del-GCN', 'Del-MHA', 'Del-FastKAN']
auc_values = [97.34, 95.13, 94.06, 93.65, 95.84]

# 创建柱状图
plt.figure(figsize=(6, 5))
bars = plt.bar(models, auc_values, width=0.45, color=['#ff3f53', '#0093ff', '#a148ff', '#ffc23f', '#75d163'])

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), va='bottom', ha='center')

# 设置图表标题和标签
# plt.title('(A)')
plt.ylabel('AUPR (%)', fontsize=12)
plt.ylim(90, 98)


plt.savefig('ablatioin_AUPR.pdf', format='pdf', bbox_inches='tight')
# 显示图表
plt.show()
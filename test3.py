import matplotlib.pyplot as plt

# 示例字典，每个键对应两个值
data = {
    'A': (10, 20),
    'B': (15, 25),
    'C': (20, 30),
    'D': (25, 35),
    'E': (30, 40)
}

# 提取键和值
keys = list(data.keys())
values1 = [data[key][0] for key in keys]
values2 = [data[key][1] for key in keys]

# 设置柱状图的位置
x = range(len(keys))
bar_width = 0.35

# 创建柱状图
plt.bar(x, values1, width=bar_width, color='skyblue', label='Value 1')
plt.bar([i + bar_width for i in x], values2, width=bar_width, color='orange', label='Value 2')

# 设置横坐标标签
plt.xticks([i + bar_width / 2 for i in x], keys)

# 添加标题和标签
plt.title('双柱状图示例')
plt.xlabel('类别')
plt.ylabel('值')

# 添加图例
plt.legend()

# 显示图形
plt.show()
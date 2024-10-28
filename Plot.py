# 加载保存的 list
import torch
import matplotlib.pyplot as plt

split_point = 19
loaded_tensor_list = torch.load(f"results/tensor-{split_point}.pt")

# 打印加载的 list
for i, tensor in enumerate(loaded_tensor_list):
    x = tensor.flatten().cpu().numpy()
    plt.figure(i)
    plt.hist(x, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    plt.xlabel('bins')
    plt.ylabel('density')
    plt.title(f'round : {i * 5}, split point : {split_point}')
    plt.show()
    if i == 5:
        break

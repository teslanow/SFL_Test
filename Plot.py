# 加载保存的 list
import pickle

import math
import torch
import matplotlib.pyplot as plt
#
# split_point = 19
# loaded_tensor_list = torch.load(f"results/tensor-{split_point}.pt")
#
# # 打印加载的 list
# for i, tensor in enumerate(loaded_tensor_list):
#     x = tensor.flatten().cpu().numpy()
#     plt.figure(i)
#     plt.hist(x, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
#     plt.xlabel('bins')
#     plt.ylabel('density')
#     plt.title(f'round : {i * 5}, split point : {split_point}')
#     plt.show()
#     if i == 5:
#         break

def calculate_divergence(model_st1, model_st2):
    total_norm = 0.0
    for key in model_st1:
        if key in model_st2:
            diff = (model_st1[key] - model_st2[key]).float()
            param_norm = diff.norm(2)
            total_norm += (param_norm ** 2).item()
    total_norm = total_norm ** 0.5
    return total_norm
def random_weigth_divergence():
    all_total_norm = []
    for idx in range(1, 101, 1):
        # model_st1 = torch.load(f'results/tmp_models3/fedavg_pretrain/{idx}.pt', map_location=torch.device('cuda:0'))
        # model_st2 = torch.load(f'results/tmp_models3/CentralLearning/{idx}.pt', map_location=torch.device('cuda:0'))
        model_st1 = torch.load(f'/data/zhongxiangwei/tmp/SFL_Test2/tmp_models_random_initial/tmp_models/fedavg_pretrain/{idx}.pt', map_location=torch.device('cuda:0'))
        model_st2 = torch.load(f'/data/zhongxiangwei/tmp/SFL_Test2/tmp_models_random_initial/tmp_models/CentralLearning2/{idx}.pt', map_location=torch.device('cuda:0'))
        total_norm = 0.0
        for key in model_st1:
            if key in model_st2:
                diff = (model_st1[key] - model_st2[key]).float()
                param_norm = diff.norm(2)
                total_norm += (param_norm ** 2).item()
        total_norm = total_norm ** 0.5
        # all_total_norm.append(total_norm)
        ct_norm = 0.0
        for key in model_st2:
            if key in model_st1:
                value = model_st2[key].float().norm(2) ** 2
                ct_norm += value.item()
        ct_norm = ct_norm ** 0.5
        all_total_norm.append(total_norm / ct_norm)
    with open('results/total_norm_random.pkl', 'wb') as f:
        pickle.dump(all_total_norm, f)

def pretrained_weight_divergence():
    all_total_norm = []
    for idx in range(1, 101, 1):
        model_st1 = torch.load(f'/data/zhongxiangwei/tmp/SFL_Test2/tmp_models3/fedavg_pretrain/{idx}.pt', map_location=torch.device('cuda:0'))
        model_st2 = torch.load(f'/data/zhongxiangwei/tmp/SFL_Test2/tmp_models3/CentralLearning/{idx}.pt', map_location=torch.device('cuda:0'))
        total_norm = 0.0
        for key in model_st1:
            if key in model_st2:
                diff = (model_st1[key] - model_st2[key]).float()
                param_norm = diff.norm(2)
                total_norm += (param_norm ** 2).item()
        total_norm = total_norm ** 0.5
        # all_total_norm.append(total_norm)
        ct_norm = 0.0
        for key in model_st2:
            if key in model_st1:
                value = model_st2[key].float().norm(2) ** 2
                ct_norm += value.item()
        ct_norm = ct_norm ** 0.5
        all_total_norm.append(total_norm / ct_norm)
    with open('results/total_norm_pretrained.pkl', 'wb') as f:
        pickle.dump(all_total_norm, f)

def plot():
    with open('results/total_norm_random.pkl', 'rb') as f:
        all_total_norm_1 = pickle.load(f)
    with open('results/total_norm_pretrained.pkl', 'rb') as f:
        all_total_norm_2 = pickle.load(f)
    print(all_total_norm_1)
    x1 = range(len(all_total_norm_1))
    y1 = all_total_norm_1
    plt.figure(figsize=(10, 6))
    x2 = range(len(all_total_norm_2))
    y2 = all_total_norm_2
    # plt.scatter(x, y)
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlim([0, 50])
    plt.legend(['random', 'pretrained'])
    plt.show()

# random_weigth_divergence()
# pretrained_weight_divergence()
plot()
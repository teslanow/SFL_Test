import pickle
import os
import matplotlib.pyplot as plt
import torch


def plot_bar(x, y, title):
    plt.bar(x, y)
    plt.title(title)
    plt.show()
    plt.close()

def plot_bar_2(x, y1, y2, title, y1_name, y2_name):
    xxx = range(len(x))
    bar_width = 0.35
    plt.bar(xxx, y1, width=bar_width, color='skyblue', label=y1_name)
    plt.bar([i + bar_width for i in xxx], y2, width=bar_width, color='orange', label=y2_name)
    plt.xticks([i + bar_width / 2 for i in xxx], x)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()


path = '/data/zhongxiangwei/tmp/SFL_Test2/test2/'
# path = '/data/zhongxiangwei/tmp/SFL_Test2/tmp_models_random_initial/tmp_models/norm'
device = 'cuda:0'
for idx, cur_round in enumerate(range(1, 51, 5)):
# for cur_round in range(1, 51, 5):
    print('cur_round', cur_round)
    round_path = os.path.join(path, str(cur_round))
    with open(os.path.join(round_path, 'all.pkl'), 'rb') as f:
        total_avg_grad = pickle.load(f)
    client_avg_grad = {}
    for root, dirs, files in os.walk(round_path):
        for file in files:
            if file != 'all.pkl':
                sp = file.split('.')
                client_id = int(sp[0])
                with open(os.path.join(round_path, file), 'rb') as f:
                    avg_grad = pickle.load(f)
                    client_avg_grad[client_id] = avg_grad
    for name, param in total_avg_grad.items():
        avg_grad = param
        avg_grad = avg_grad.to(device)
        name_norm = {}
        name_dot = {}
        for client_id in client_avg_grad.keys():
            client_all_grad = client_avg_grad[client_id]
            if name in client_all_grad:
                client_grad = client_all_grad[name]
                client_grad = client_grad.to(device)
                # 计算2-norm
                name_dot[client_id] = (avg_grad - client_grad).norm(2).item()
                # name_dot[client_id] = torch.dot(avg_grad, client_grad).item()
                name_norm[client_id] = (avg_grad.norm(2)).item()
        # name_norm['all'] = avg_grad.norm(2).item()
        # name_dot['all'] = avg_grad.norm(2).item()
        x, y1, y2 = [], [], []
        for key, val in name_norm.items():
            x.append(str(key))
            y1.append(name_norm[key])
            y2.append(name_dot[key])
        plot_bar_2(x, y1, y2, f'round-{cur_round}-name-{name}', 'term 3', 'term 4')
        # plot_bar(x, y1, f'round-{cur_round}-name-{name}')
        # exit()
    if idx == 1:
        break


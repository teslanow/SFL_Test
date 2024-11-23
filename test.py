# 测试数据是不是真的non-iid
import argparse
import copy
import os
import pickle
import random
from collections import OrderedDict
import torch
from torchvision import datasets
from tqdm import tqdm
from ptflops import get_model_complexity_info
from fedlab.models.CommModels import create_model_full, model_density_per_layer
from fedlab.utils.utils import set_seed, load_default_transform
from torch.utils.data import DataLoader



def parse_args():
    # init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='cifar10')
    parser.add_argument('--model_type', type=str, default='Resnet34')
    parser.add_argument('--client_batch_size', type=int, required=True)
    parser.add_argument('--round', type=int, default=250, help='communication round')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--client_data_path', type=str, required=True)
    parser.add_argument('--class_num', type=int, default=10, required=True)
    parser.add_argument('--models_path', type=str, help='需要先通过FL算法得到每一round的model，放到models_path文件夹下', required=True)
    parser.add_argument('--path_sample_clients', type=str, required=True)
    return parser.parse_args()

def statistical_gradient(model: torch.nn.Module, data_loader, optimizer, device, criterion):
    all_grads = OrderedDict()
    model.train()
    model.to(device)
    # for data, target in tqdm(data_loader):
    total_sample_num = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        sample_num = data.size(0)
        total_sample_num += sample_num
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in all_grads:
                    all_grads[name] = []
                all_grads[name].append(param.grad.clone().detach().view(-1) * sample_num)
        del output
        torch.cuda.empty_cache()
        break
    avg_grads = {}
    for name, grads in all_grads.items():
        avg_grads[name] = torch.stack(grads).sum(dim=0) / total_sample_num
    all_grads.clear()
    return avg_grads

def sample_gradient(model: torch.nn.Module, data, target, optimizer, device, criterion):
    all_grads = OrderedDict()
    model.train()
    model.to(device)
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in all_grads:
                all_grads[name] = None
            all_grads[name] = param.grad.clone().detach().view(-1)
    del output
    torch.cuda.empty_cache()
    return all_grads

def calculate_norm(tensor, p):
    return tensor.norm(p).item()

def load_train_dataset_from_clients(client_id, path):
    return torch.load(os.path.join(path, 'fedlab', 'train', "data{}.pkl".format(client_id)))

def load_train_dataset_from_clients_total(client_ids, path):
    all_dataset = []
    for client_id in client_ids:
        dataset = torch.load(os.path.join(path, 'fedlab', 'train', "data{}.pkl".format(client_id)))
        all_dataset.append(dataset)
    return torch.utils.data.ConcatDataset(all_dataset)

def main():
    set_seed()
    args = parse_args()
    model = create_model_full(args.model_type, (args.class_num, True))
    optim= torch.optim.SGD(model.parameters(), lr=0.001)
    transf = load_default_transform(args.dataset_type)
    # overall_dataset = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transf)
    # overall_loader = DataLoader(overall_dataset, batch_size=256, shuffle=True)
    overall_dataset = load_train_dataset_from_clients_total(range(100), args.data_path)
    overall_loader = DataLoader(overall_dataset, batch_size=args.client_batch_size, shuffle=True)
    # 开始训练
    device = "cuda:0"
    # 加载fl的client序列
    with open(args.path_sample_clients, 'rb') as f:
        all_sample_clients = pickle.load(f)
        print(all_sample_clients)
    cri = torch.nn.CrossEntropyLoss()
    for idx, cur_round in enumerate(range(1, 51, 5)):
        round_path = f'/data/zhongxiangwei/tmp/SFL_Test2/test2/{cur_round}'
        # round_path = f'/data/zhongxiangwei/tmp/SFL_Test2/tmp_models_random_initial/tmp_models/norm/{cur_round}'
        if not os.path.exists(round_path):
            os.makedirs(round_path)
        model_st = torch.load(os.path.join(args.models_path, str(cur_round) + '.pt'))
        model.load_state_dict(model_st)
        print("train on total dataset")
        avg_grads = statistical_gradient(model, overall_loader, optim, device, criterion=cri)
        with open(os.path.join(round_path, 'all.pkl'), 'wb') as f:
            pickle.dump(avg_grads, f)
        del avg_grads
        # sample_clients = all_sample_clients[cur_round - 1]
        sample_clients = all_sample_clients[cur_round]
        for client_id in sample_clients:
            print("train on {client_id}".format(client_id=client_id))
            client_dataset = load_train_dataset_from_clients(client_id, args.client_data_path)
            # client_loader = DataLoader(client_dataset, batch_size=args.client_batch_size, shuffle=True)
            client_loader = DataLoader(client_dataset, batch_size=256, shuffle=True)
            # for data, target in client_loader:
            #     per_sample_grad = sample_gradient(model, data, target, optim, device, criterion=cri)
            #     # 计算norm
            #     total_norm = 0.0
            #     for name, param in per_sample_grad.items():
            #         total_norm += (param.norm(2) ** 2).item()
            #     print(f"local gradient 2-norm: {total_norm}")
            #     total_dots = 0.0
            #     for name, param in per_sample_grad.items():
            #         if name in avg_grads:
            #             total_dots += torch.dot(param, avg_grads[name]).item()
            #     print(f"norm: {total_norm}, dot: {total_dots}")
            avg_grads = statistical_gradient(model, client_loader, optim, 'cuda:1', cri)
            with open(os.path.join(round_path, f'{client_id}.pkl'), 'wb') as f:
                pickle.dump(avg_grads, f)
            del avg_grads
        # break

def main2():
    set_seed()
    args = parse_args()
    model = create_model_full(args.model_type, (args.class_num, True))
    optim= torch.optim.SGD(model.parameters(), lr=0.001)
    overall_dataset = load_train_dataset_from_clients_total(range(100), args.data_path)
    overall_loader = DataLoader(overall_dataset, batch_size=256, shuffle=True)
    # 开始训练
    device = "cuda:0"
    # 加载fl的client序列
    cri = torch.nn.CrossEntropyLoss()
    model_st = torch.load("results/tmp_models/fedavg_pretrain/1.pt")
    model.load_state_dict(model_st)
    print("train on total dataset")
    avg_grads = statistical_gradient(model, overall_loader, optim, device, criterion=cri)
    with open(os.path.join("results/tmp_models/fedavg_pretrain/", 'all.pkl'), 'wb') as f:
        pickle.dump(avg_grads, f)
    del avg_grads
    sample_clients = random.sample(range(100), 10)
    for client_id in sample_clients:
        print("train on {client_id}".format(client_id=client_id))
        client_dataset = load_train_dataset_from_clients(client_id, args.client_data_path)
        client_loader = DataLoader(client_dataset, batch_size=256, shuffle=True)
        avg_grads = statistical_gradient(model, client_loader, optim, device, cri)
        with open(os.path.join("results/tmp_models/fedavg_pretrain/", f'client-{client_id}.pkl'), 'wb') as f:
            pickle.dump(avg_grads, f)
        del avg_grads
        # break

def main3():
    set_seed()
    args = parse_args()
    model = create_model_full(args.model_type, (args.class_num, True))
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, backend='pytorch',print_per_layer_stat=True, verbose=True)
    print(macs)

if __name__ == '__main__':
    main3()
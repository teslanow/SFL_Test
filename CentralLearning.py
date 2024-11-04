import argparse
import os
import pickle

import torch
from torchvision import datasets
from tqdm import tqdm

from fedlab.models.CommModels import create_model_full
from fedlab.utils.WandbWrapper import wandbInit, wandbLogWrap
from fedlab.utils.functional import evaluate
from fedlab.utils.utils import set_seed, load_default_transform
import torch.functional as F


def wandb_config(args):
    return {
        "lr": args.lr,
        "architecture": args.model_type,
        "batch_size": args.batch_size,
        "local_epoch": args.local_epoch,
        "dataset": args.dataset_type,
        "round": args.round,
        "pretrained": args.pretrained,
        "step_size": args.step_size,
        "gamma": args.gamma,
    }

def parse_args():
    # init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='cifar10')
    parser.add_argument('--model_type', type=str, default='Resnet34')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--round', type=int, default=250, help='communication round')
    parser.add_argument('--local_epoch', type=int, default=1, help='local iteration')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_cuda', action="store_false", default=True)
    parser.add_argument('--expname', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrained', action='store_true', default=False, help='提供该参数，则表示开启预训练')
    parser.add_argument('--class_num', type=int, default=10, required=True)
    parser.add_argument('--initial_model_path', type=str, default='')
    return parser.parse_args()

def load_train_dataset_from_clients(client_ids, path):
    all_dataset = []
    for client_id in client_ids:
        dataset = torch.load(os.path.join(path, 'fedlab', 'train', "data{}.pkl".format(client_id)))
        all_dataset.append(dataset)
    return torch.utils.data.ConcatDataset(all_dataset)

def main(save=False):
    # set_seed()
    args = parse_args()
    config = wandb_config(args)
    # wandbInit(args, "CentralLearning", config)
    model = create_model_full(args.model_type, (args.class_num, args.pretrained))
    if hasattr(args, 'initial_model_path') and args.initial_model_path != '':
        print(args.initial_model_path)
        model.load_state_dict(torch.load(args.initial_model_path))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    transf = load_default_transform(args.dataset_type)
    test_dataset = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transf)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

    # 开始训练
    device = args.device
    criterion = torch.nn.CrossEntropyLoss()

    # 加载fl的client序列
    with open('results/all_sample_clients.pkl', 'rb') as f:
        all_sample_clients = pickle.load(f)
        print(all_sample_clients)
    for cur_round in range(args.round):
        sample_clients = all_sample_clients[cur_round]
        # 加载数据
        train_dataset = load_train_dataset_from_clients(sample_clients, args.data_path)
        train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        model.train()
        model.to(device)
        for epoch in range(args.local_epoch):
            for _, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        sched.step()
        loss_, acc_ = evaluate(model, criterion, test_loader)
        print(
            f"Round [{cur_round + 1 }/{args.round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100 * acc_:.3f}%"
        )
        wandbLogWrap({
            "Round": cur_round + 1,
            "Loss": loss_,
            "Acc": acc_,
        })
        if save and (cur_round + 1) % 1 == 0:
            path = "results/tmp_models/CentralLearning/"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), os.path.join(path, f"{(cur_round + 1) * args.local_epoch}.pt"))

if __name__ == '__main__':
    main(save=False)

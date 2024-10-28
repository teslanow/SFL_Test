import argparse
import copy

import torch
import torchvision
from tqdm import tqdm

from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
from fedlab.utils.WandbWrapper import *
from fedlab.models.CommModels import create_model_full, create_model_instance_SL
from fedlab.utils.functional import AverageMeter, evaluate
from fedlab.utils.utils import load_default_transform, set_seed, prRed, prGreen
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='To explore which compression methods works')
    parser.add_argument('--dataset_type', type=str, default='cifar100')
    parser.add_argument('--class_num', type=int, required=True)
    parser.add_argument('--model_type', type=str, default='Resnet34')
    # 现在实验已经把cifar100按照alpha=0.5划分为50份，id从0到49使用相应的client数据，id=50使用total数据
    parser.add_argument('--data_id', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--round', type=int, default=250, help='how many round to train')
    parser.add_argument('--local_epoch', type=int, default=1, help='local iteration')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_cuda', action="store_false", default=True)
    parser.add_argument('--pretrained', action='store_true', default=False, help='提供该参数，则表示开启预训练')
    parser.add_argument('--device', type=str, default='cuda:0')
    # 从哪里开始划分
    parser.add_argument('--split_point', type=int, required=True)
    parser.add_argument('--compress_method', type=str)
    # 如果使用quantizer
    parser.add_argument('--bits', type=int, default=32)
    # 如果使用top-k
    parser.add_argument('--k_ratio', type=float, default=1.0)
    # 如果使用frequency
    parser.add_argument('--L_ratio', type=float, default=1.0, help='ratio of Low frequency part')
    return parser.parse_args()

def wandb_config(args):
    return {
        "lr": args.lr,
        "architecture": args.model_type,
        "batch_size": args.batch_size,
        "local_epoch": args.local_epoch,
        "compress_method": args.compress_method,
        "dataset": args.dataset_type,
        "round": args.round,
        "pretrained": args.pretrained,
        "bits": args.bits,
        "k_ratio": args.k_ratio,
        "L_ratio": args.L_ratio,
    }

def evaluate_1(bottom_model, top_model, test_loader, device, criterion):
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        bottom_model.eval()
        top_model.eval()
        for inputs, labels in test_loader:
            batch_size = len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            output1 = bottom_model(inputs)
            outputs = top_model(output1)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            loss_.update(loss.item(), batch_size)
            acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)

    return loss_.avg, acc_.avg

def compress_activation(x, compress_method:str, k_ratio=1.0):
    """
    对x进行压缩，x需要被detach过
    Args:
        x: activation
        compress_method: normal，等

    Returns:

    """
    with torch.no_grad():
        if compress_method == "normal":
            return x
        elif compress_method == "topk":
            pass
        else:
            raise NotImplementedError

def main():
    set_seed()
    args = parse_args()
    config = wandb_config(args)
    # wandbInit(args, "CompressCT", config)
    bottom_model, top_model = create_model_instance_SL(args.model_type, {
        "split_point" : args.split_point,
        "class_num" : args.class_num,
        "pretrained" : args.pretrained,
    })

    test_model = torchvision.models.resnet34(pretrained= args.pretrained)
    test_model.fc = torch.nn.Linear(test_model.fc.in_features, args.class_num)
    optimizer = torch.optim.SGD(test_model.parameters(), lr=args.lr)


    transf = load_default_transform(args.dataset_type)
    bottom_optimizer = torch.optim.SGD(bottom_model.parameters(), args.lr)
    top_optimizer = torch.optim.SGD(top_model.parameters(), args.lr)
    bottom_scheduler = torch.optim.lr_scheduler.StepLR(bottom_optimizer, step_size=5, gamma=0.9)
    top_scheduler = torch.optim.lr_scheduler.StepLR(top_optimizer, step_size=5, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    # 加载数据
    # dataset = PartitionCIFAR(
    #     root=args.data_path,
    #     path=os.path.join(args.data_path, "fedlab"),
    #     dataname=args.dataset_type,
    #     num_clients=args.total_clients,
    #     partition='dirichlet',
    #     dir_alpha=0.5,
    #     seed=1234,
    #     transform=transf,
    #     preprocess=False
    # )
    # if args.data_id < 50:
    #     train_data_loader = dataset.get_dataloader(args.data_id, args.batch_size)
    # else:
    train_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transf)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transf)
    test_data_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for cur_round in range(args.round):
        for cur_epoch in range(args.local_epoch):
            test_model.train()
            test_model.to(args.device)
            # bottom_model.train()
            # top_model.train()
            # bottom_model.to(args.device)
            # top_model.to(args.device)
            for batch_idx, (data, target) in enumerate(tqdm(train_data_loader, desc='Training')):
                data = data.to(args.device)
                target = target.to(args.device)
                output = test_model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss, acc = evaluate(test_model, criterion, test_data_loader)
                # output = bottom_model(data)
                # smashed_data = output.clone().detach()
                # # 压缩
                # # new_x = compress_activation(output, args.compress_method)
                # smashed_data.requires_grad = True
                # output1 = top_model(smashed_data)
                # loss = criterion(output1, target)
                # top_optimizer.zero_grad()
                # loss.backward()
                # top_optimizer.step()
                # # bottom_model的后向
                # grad_smashed_data = smashed_data.grad.clone().detach()
                # bottom_optimizer.zero_grad()
                # output.backward(grad_smashed_data)
                # bottom_optimizer.step()
            # bottom_scheduler.step()
            # top_scheduler.step()
            # loss, acc = evaluate(bottom_model, top_model, test_data_loader, args.device, criterion)
            prRed(f"loss : {loss}, acc : {acc}")
    wandbFinishWrap()

main()


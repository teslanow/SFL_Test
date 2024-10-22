import argparse
import json
import signal
from datetime import datetime
import logging
import os
import random

import numpy as np
import torch
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter
from typing import List
# from Common.ClientProperties import ClientPropertyManager
from torchvision import transforms
from fedlab.utils.WandbWrapper import wandbFinishWrap
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def parse_args():
    # init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='cifar10')
    parser.add_argument('--model_type', type=str, default='Resnet34')
    parser.add_argument('--total_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_pattern', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.993)
    parser.add_argument('--min_lr', type=float, default=0.005)
    parser.add_argument('--round', type=int, default=250, help='communication round')
    parser.add_argument('--local_epoch', type=int, default=1, help='local iteration')
    parser.add_argument('--momentum', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_cuda', action="store_false", default=True)
    parser.add_argument('--expname', type=str, required=True)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--sys_conf_path', type=str, default='ExpConfig/System_conf.yml')
    # 指定一个同步轮训练中多少比例的clients能完成训练
    parser.add_argument('--finish_ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrained', action='store_true', default=False, help='提供该参数，则表示开启预训练')

    return parser.parse_args()

def get_client_logger(args, rank):
    RESULT_PATH = os.getcwd() + '/clients/' + args.expname + '/'

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)

    logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
    logger.setLevel(logging.INFO)

    filename = RESULT_PATH + os.path.basename(__file__).split('.')[0] + '_' + str(int(rank)) + '.log'
    fileHandler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger

def set_recorder_and_logger(args):
    RESULT_PATH = os.getcwd() + '/server/'

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)

    cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    recorder: SummaryWriter = SummaryWriter(os.path.join('runs', args.expname, f"{args.finish_ratio}_{args.data_pattern}_{cur_time}"))

    logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
    logger.setLevel(logging.INFO)

    # now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    filename = RESULT_PATH + args.expname + "_" + os.path.basename(__file__).split('.')[0] + '.log'
    fileHandler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # logger.info(f"corresponding runtime file: {args.finish_ratio}_{args.data_pattern}_{cur_time}")

    return recorder, logger

class Sys_conf:
    def __init__(self):
        self.client_compute_density : List[float]  = None
        self.server_compute_density : List[float] = None
        self.client_communicate_bandwidth : List[float] = None

def load_system_config(path) -> Sys_conf:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    sys_conf = Sys_conf()
    for k, v in config.items():
        setattr(sys_conf, k, v)
    return sys_conf


def set_seed():
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

def prepare_running():
    args = parse_args()
    # recorder, logger = set_recorder_and_logger(args)
    # path = os.getcwd()
    # print(path)
    # path = path + "//" + "result_recorder"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # path = path + "//" + now + "_record.txt"
    # result_out = open(path, 'w+')
    # result_out.write(f"\ncorresponding dir : {recorder.get_logdir()}\n")
    # json.dump(args.__dict__, result_out, indent=4)
    # print(args.__dict__, file=result_out)
    # result_out.write('\n')
    # result_out.write("epoch_idx, total_time, total_bandwith, total_resource, acc, test_loss")
    # result_out.write('\n')

    recorder, logger, result_out = None, None, None
    # client_manager = ClientPropertyManager('ExpConfig/vehicle_device_capacity')
    # 设置signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGKILL, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    return args, recorder, logger, None, result_out

def signal_handler(sig, frame):
    print(f'You pressed {sig}!')
    # 在这里执行清理操作
    wandbFinishWrap()
    exit()

def compare_layer_difference(model1, model2, device):
    model1.to(device)
    model2.to(device)
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    name_to_norm_dict = dict()
    for name, param1 in params1.items():
        param2 = params2[name]
        diff = param1 - param2
        diff_squared_sum = torch.sum(diff ** 2)
        name_to_norm_dict[name] = diff_squared_sum.item()
    return name_to_norm_dict

def load_default_transform(dataset_type, train=False):
    if dataset_type == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
        else:
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    elif dataset_type == 'cifar100':
        # reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
        else:
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
    else:
        raise NotImplementedError

    return dataset_transform
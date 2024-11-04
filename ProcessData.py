import torch

from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
import os

from fedlab.models.CommModels import create_model_full
from fedlab.utils.utils import set_seed, parse_args, load_default_transform
# transf = load_default_transform("cifar100")
# dataset = PartitionCIFAR(
#     root="/data/zhongxiangwei/data/CIFAR100",
#     path=os.path.join("/data/zhongxiangwei/data/CIFAR100", "fedlab"),
#     dataname="cifar100",
#     num_clients=50,
#     partition='dirichlet',
#     dir_alpha=0.5,
#     seed=1234,
#     transform=transf,
#     preprocess=True
# )


# transf = load_default_transform("cifar10")
# dataset = PartitionCIFAR(
#     root="/data/zhongxiangwei/data/CIFAR10",
#     path=os.path.join("/data/zhongxiangwei/data/CIFAR10", "fedlab"),
#     dataname="cifar10",
#     num_clients=100,
#     partition='dirichlet',
#     dir_alpha=0.5,
#     seed=1234,
#     transform=transf,
#     preprocess=True
# )
# dataset.preprocess()


# model = create_model_full('resnet34', (10, False))
# torch.save(model.state_dict(), 'results/resnet34-class-10-no-pretrained.pt')
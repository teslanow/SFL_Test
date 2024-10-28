from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
import os
from fedlab.utils.utils import set_seed, parse_args, load_default_transform
transf = load_default_transform("cifar100")
dataset = PartitionCIFAR(
    root="/data/zhongxiangwei/data/CIFAR100",
    path=os.path.join("/data/zhongxiangwei/data/CIFAR100", "fedlab"),
    dataname="cifar100",
    num_clients=50,
    partition='dirichlet',
    dir_alpha=0.5,
    seed=1234,
    transform=transf
)
dataset.preprocess()
import os.path
import random

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipelineWithCompression
from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
from fedlab.models.CommModels import create_model_instance_SL
from fedlab.utils.utils import set_seed, parse_args, load_default_transform
from fedlab.utils.WandbWrapper import wandbInit, wandbFinishWrap


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
        "split_point": args.split_point,
    }

set_seed()
args = parse_args()
config = wandb_config(args)
wandbInit(args, "FedAvg_Compress_2", config)
bottom_model, top_model = create_model_instance_SL(args.model_type, {
    "split_point" : args.split_point,
    "class_num" : args.class_num,
    "pretrained" : args.pretrained,
})
# server
handler = SyncServerHandler(
    model=top_model, global_round=args.round, num_clients=args.total_clients, sample_ratio=args.sample_ratio, cuda=True, device=args.device, bottom_model=bottom_model
)

# client
trainer = SGDSerialClientTrainer(bottom_model, args.total_clients, cuda=True, device=args.device, server_model=top_model)
transf = load_default_transform(args.dataset_type)
# 先调用ProcessData产生client所需的数据
dataset = PartitionCIFAR(
    root=args.data_path,
    path=os.path.join(args.data_path, "fedlab"),
    dataname=args.dataset_type,
    num_clients=args.total_clients,
    partition='dirichlet',
    dir_alpha=0.5,
    seed=1234,
    transform=transf,
    preprocess=False
)
# dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(args.local_epoch, args.batch_size, args.lr)

handler.num_clients = args.total_clients
handler.setup_dataset(dataset)
compression_args = {
    "compress_method" : args.compress_method,
    "k_ratio" : args.k_ratio,
    "bits" : args.bits,
}
# main
pipeline = StandalonePipelineWithCompression(handler, trainer)
pipeline.main(compression_args)
wandbFinishWrap()

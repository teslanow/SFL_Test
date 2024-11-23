import copy
import os.path
import random

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
from fedlab.models.CommModels import create_model_full, model_density_per_layer
from fedlab.utils.utils import set_seed, parse_args, load_default_transform, load_input_tensor_type
from fedlab.utils.WandbWrapper import wandbInit, wandbFinishWrap
import fedlab.utils.simulate_time
from fedlab.utils.System_conf import set_cur_system_hetero, get_cur_system_hetero

def wandb_config(args):
    return {
        "lr": args.lr,
        "architecture": args.model_type,
        "batch_size": args.batch_size,
        "local_epoch": args.local_epoch,
        "dataset": args.dataset_type,
        "round": args.round,
        "pretrained": args.pretrained,
        "sample_ratio": args.sample_ratio,
        "total_clients": args.total_clients,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "system_hetero" : args.system_hetero

    }

set_seed()
args = parse_args()
config = wandb_config(args)
set_cur_system_hetero(args.system_hetero)
print("当前system_hetero : ", args.system_hetero)
# wandbInit(args, "FedAvg_Time_1", config)
model = create_model_full(args.model_type, (args.class_num, args.pretrained))
handler = SyncServerHandler(
    model=model, global_round=args.round, num_clients=args.total_clients, sample_ratio=args.sample_ratio, device=args.device, cuda=True
)
# client
trainer = SGDSerialClientTrainer(model, args.total_clients, cuda=True, device=args.device)
transf = load_default_transform(args.dataset_type)
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
trainer.setup_optim(args.local_epoch, args.batch_size, args.lr, args.step_size, args.gamma)

handler.num_clients = args.total_clients
handler.setup_dataset(dataset)
# main
pipeline = StandalonePipeline(handler, trainer)
pipeline.set_clients_properties('ExpConfig/vehicle_device_capacity', args.total_clients)
# for _ in range(10):
#     print(pipeline.clientPropertyManager.client_profiles[random.randint(0, len(pipeline.clientPropertyManager.client_profiles) - 1)])

# 获取model的density
input_tensor_shape = load_input_tensor_type(args.dataset_type)
# macs, params = model_density_per_layer(copy.deepcopy(model), input_tensor_shape)
macs, params = model_density_per_layer(copy.deepcopy(model), (3, 224, 224))
pipeline.main(save=True, model_density=macs * 3)
wandbFinishWrap()

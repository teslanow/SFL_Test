import copy
import os.path
import random

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipelineWithFreeze
from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
from fedlab.models.CommModels import create_model_full, model_density_per_layer
from fedlab.utils.System_conf import set_cur_system_hetero
from fedlab.utils.utils import set_seed, parse_args, load_default_transform, load_input_tensor_type
from fedlab.utils.WandbWrapper import wandbInit, wandbFinishWrap
import fedlab.utils.simulate_time
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
        "update_method": args.update_method,
        "num_bk": args.num_bk if hasattr(args, 'num_bk') else 0,
        "percentile": args.percentile if hasattr(args, 'percentile') else 50,
        "pd_sc_step": args.pd_sc_step if hasattr(args, 'pd_sc_step') else 10,
    }

set_seed()
args = parse_args()
config = wandb_config(args)
set_cur_system_hetero(args.system_hetero)
# wandbInit(args, "Freeze-3", config)
model = create_model_full(args.model_type, (args.class_num, args.pretrained))
# server
handler = SyncServerHandler(
    model=model, global_round=args.round, num_clients=args.total_clients, sample_ratio=args.sample_ratio, cuda=True, device=args.device
)

# client
trainer = SGDSerialClientTrainer(model, args.total_clients, cuda=True, device=args.device)
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
trainer.setup_optim(args.local_epoch, args.batch_size, args.lr, args.step_size, args.gamma)

handler.num_clients = args.total_clients
handler.setup_dataset(dataset)
# main
pipeline = StandalonePipelineWithFreeze(handler, trainer)
pipeline.set_clients_properties('ExpConfig/vehicle_device_capacity', args.total_clients)
# 设置freeze args
if args.update_method == 'static':
    freeze_args = args.num_bk
elif args.update_method == 'pd_sc':
    # progressive decreasing layer freezing, same for all clients:
    freeze_args = (args.percentile, args.pd_sc_step)
elif args.update_method == 'random_sync':
    # 每个client随机分配冻结的层，在训练开始时确定一次，后续不再发生变化
    # 先获取可冻结的参数的总数
    model = trainer.model
    num_freezable = len(list(model.parameters()))
    freeze_args = num_freezable
elif args.update_method == 'selective':
    freeze_args = None
else:
    raise NotImplementedError
# 获取model的density
# input_tensor_shape = load_input_tensor_type(args.dataset_type)
macs, params = model_density_per_layer(copy.deepcopy(model), (3, 224, 224))
pipeline.main(freeze_args, freeze_method=args.update_method, forward_model_density=macs)
wandbFinishWrap()

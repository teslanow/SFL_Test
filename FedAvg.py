import os.path
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
from fedlab.models.CommModels import create_model_full
from fedlab.utils.utils import set_seed, parse_args, load_default_transform
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
        "total_clients": args.total_clients,
        "step_size": args.step_size,
        "gamma": args.gamma,
    }

set_seed()
args = parse_args()
config = wandb_config(args)
wandbInit(args, "FedAvg_3", config)
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
pipeline.main(save=True)
wandbFinishWrap()

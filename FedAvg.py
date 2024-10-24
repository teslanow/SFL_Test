import os.path
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.contrib.dataset.partitioned_cifar import PartitionCIFAR
from fedlab.models.CommModels import create_model_full
from fedlab.utils.utils import set_seed, parse_args, load_default_transform
from fedlab.utils.WandbWrapper import wandbInit, wandbFinishWrap

set_seed()
args = parse_args()
wandbInit(args)
model = create_model_full(args.model_type, (100, args.pretrained))
# server
handler = SyncServerHandler(
    model=model, global_round=args.round, num_clients=args.total_clients, sample_ratio=args.sample_ratio
)

# client
trainer = SGDSerialClientTrainer(model, args.total_clients, cuda=True)
transf = load_default_transform(args.dataset_type)
dataset = PartitionCIFAR(
    root=args.data_path,
    path=os.path.join(args.data_path, "fedlab"),
    dataname=args.dataset_type,
    num_clients=args.total_clients,
    partition='dirichlet',
    dir_alpha=0.5,
    seed=1234,
    transform=transf
)
dataset.preprocess()

trainer.setup_dataset(dataset)
trainer.setup_optim(args.local_epoch, args.batch_size, args.lr)

handler.num_clients = args.total_clients
handler.setup_dataset(dataset)
# main
pipeline = StandalonePipeline(handler, trainer)
pipeline.main()
wandbFinishWrap()

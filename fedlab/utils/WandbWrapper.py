import wandb

def wandbInit(args, project_name=None, config=None):
    if config is None:
        config = {
            "lr": args.lr,
            "architecture": args.model_type,
            "batch_size": args.batch_size,
            "local_epoch": args.local_epoch,
            "algorithm": args.expname,
            "total_clients": args.total_clients,
            "sample_ratio": args.sample_ratio,
            "dataset": args.dataset_type,
            "global_rounds": args.round,
            "data_pattern": args.data_pattern,
            "pretrained": args.pretrained,
            "update_method": args.update_method if hasattr(args, 'update_method') else None,
            "num_bk": args.num_bk if hasattr(args, 'num_bk') else None,
            "percentile": args.percentile if hasattr(args, 'percentile') else None,
            "pd_sc_step": args.pd_sc_step if hasattr(args, 'pd_sc_step') else None,
        }
    if project_name is None:
        project_name = "SFL"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        # track hyperparameters and run metadata
        config=config
    )

def wandbLogWrap(content):
    if wandb.run is not None:
        wandb.log(content)

def wandbFinishWrap():
    if wandb.run is not None:
        wandb.finish()

# static
# pd_sc
# python FedAvg_Freeze.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.02 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedAvg_no_freeze --update_method static --num_bk 92  --device cuda:3
device = [0, 1, 4, 5, 6, 7]
i = 0
for pd_sc_step in range(1, 6):
    for percentile in [10, 30, 50, 70, 90, 100]:
        print(f"python FedAvg_Freeze.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.02 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedAvg_pdsc --update_method pd_sc --pd_sc_step {pd_sc_step} --percentile {percentile} --device cuda:{device[i % len(device)]} &")
        i += 1
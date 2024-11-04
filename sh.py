# static
# pd_sc
# device = [0, 1, 2, 3, 5, 6]
# i = 0
# for num_bk in [2, 29, 41, 53, 71, 86, 92, 100]:
#     print(f"python FedAvg_Freeze.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.1 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedAvg_freeze --update_method static --class_num 100 --num_bk {num_bk}  --device cuda:{device[i % len(device)]} &")
#     i += 1

# python FedAvg_Freeze.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.02 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedAvg_no_freeze --update_method static --class_num 100 --num_bk 92  --device cuda:3
# device = [0, 1, 2, 3, 5, 6]
# i = 0
# for pd_sc_step in range(1, 6):
#     for percentile in [10, 30, 50, 70, 90, 100]:
#         print(f"python FedAvg_Freeze.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.1 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedAvg_pdsc --update_method pd_sc --pd_sc_step {pd_sc_step} --percentile {percentile} --class_num 100 --device cuda:{device[i % len(device)]} &")
#         i += 1


# device = [0, 1, 2, 3, 4]
# i = 0
# for split_point in [5, 10, 15, 19]:
#     for bits in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
#         # 量化
#         print(f"python LossyCompressCT.py --dataset_type cifar100 --model_type resnet34 --batch_size 64 --lr 0.001 --round 100 --local_epoch 1 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --split_point {split_point} --class_num 100 --data_id 50 --compress_method quantize --bits {bits} --device cuda:{device[i % len(device)]} &")
#         i += 1


# top-k
# for split_point in [5, 10, 15, 19]:
#     for k_ratio in [0.05, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35,  0.4, 0.45, 0.5]:
#         # 量化
#         print(f"python LossyCompressCT.py --dataset_type cifar100 --model_type resnet34 --batch_size 64 --lr 0.001 --round 50 --local_epoch 1 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --split_point {split_point} --class_num 100 --data_id 50 --compress_method topk --k_ratio {k_ratio} --device cuda:{device[i % len(device)]} &")
#         i += 1

# fedsplit-compress
# i = 0
# device = [0, 1, 2, 3, 5, 6]
# for split_point in [5, 10, 15, 19]:
#     for k_ratio in [0.05, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35,  0.4, 0.45, 0.5]:
#         # 量化
#         print(f"python FedAvg_Split_Compress.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.1 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedSplit_Compress --class_num 100 --compress_method topk --k_ratio {k_ratio} --device cuda:{device[i % len(device)]} --split_point {split_point} &")
#         i += 1

i = 0
device = [0, 1, 2, 3, 5, 6]
for split_point in [5, 10, 15, 19]:
    for bits in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # 量化
        print(f"python FedAvg_Split_Compress.py --dataset_type cifar100 --model_type resnet34 --total_clients 50 --batch_size 64 --lr 0.1 --round 100 --local_epoch 1 --sample_ratio 0.2 --pretrained --data_path /data/zhongxiangwei/data/CIFAR100 --expname FedSplit_Compress --class_num 100 --compress_method quantize --bits {bits} --device cuda:{device[i % len(device)]} --split_point {split_point} &")
        i += 1
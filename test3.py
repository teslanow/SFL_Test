import os
from collections import Counter
import torch
# for cid in range(100):
#     dataset = torch.load(os.path.join('/data/zhongxiangwei/data/CIFAR10/fedlab-iid', 'train', "data{}.pkl".format(cid)))
#     count = Counter(dataset.targets)
#     count = sorted(count.items(), key=lambda item: item[0])
#     print(len(dataset))
#     print(count)
from fedlab.models.CommModels import model_density_per_layer
import copy
from fedlab.models.CommModels import create_model_full, model_density_per_layer
model = create_model_full('resnet34', (10, True))
macs, params = model_density_per_layer(copy.deepcopy(model), (3, 224, 224))
print(macs, params)
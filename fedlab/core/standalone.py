# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
import copy
import random

import numpy as np
import torch
from fedlab.utils.WandbWrapper import wandbLogWrap
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .client.trainer import SerialClientTrainer
from .server.handler import ServerHandler
from ..utils.functional import AverageMeter


class StandalonePipeline(object):
    def __init__(self, handler: ServerHandler, trainer: SerialClientTrainer):
        """Perform standalone simulation process.

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer

        # initialization
        self.handler.num_clients = self.trainer.num_clients

    def main(self):
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()
            # self.handler.evaluate()

    def evaluate(self):
        loss_, acc_ = self.handler.evaluate()


def static_decision_to_update(num, model: torch.nn.Module):
    """
    离线确定哪些层该更新
    Args:
        num: 应该更新的block的数目（从后往前数）
    """
    name_list_to_update = []
    for name, param in model.named_parameters():
        name_list_to_update.append(name)
    name_list_to_update = name_list_to_update[-num:]
    return name_list_to_update

def achieve_name_list_by_indices(total_name_list, indices):
    result = []
    for index in indices:
        if index < len(total_name_list):
            result.append(total_name_list[index])
    return result

def pd_sc_to_update(pd_sc_percentile, last_state_dict, cur_state_dict, origin_name_list_to_update):
    name_list = origin_name_list_to_update
    param_change_list = []
    for name in name_list:
        last_param_gradient_norm = torch.norm(last_state_dict[name]) ** 2
        cur_param_gradient_norm = torch.norm(cur_state_dict[name]) ** 2
        param_change_list.append((torch.abs(last_param_gradient_norm - cur_param_gradient_norm) / (last_param_gradient_norm + torch.tensor(0.01))).item())
    percentile_value = np.percentile(np.array(param_change_list), pd_sc_percentile)
    for index, change in enumerate(param_change_list):
        if change >= percentile_value:
            return name_list[index:]
    return []



class StandalonePipelineWithFreeze(object):
    def __init__(self, handler: ServerHandler, trainer: SerialClientTrainer):
        """Perform standalone simulation process.

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer

        # initialization
        self.handler.num_clients = self.trainer.num_clients

        # 用于pd_sc
        self.prev_handler_model_state_dict = None

    def main(self, freeze_args, freeze_method):
        name_list_to_update = [str(name) for name, param in self.trainer.model.named_parameters()]
        total_name_list = copy.deepcopy(name_list_to_update)
        min_list_to_update = copy.deepcopy(name_list_to_update[-2:])
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package
            # 确定哪些层要freeze，哪些层要update
            if freeze_method == 'static':
                num_bk = freeze_args
                name_list_to_update = static_decision_to_update(num_bk, self.trainer.model)
            elif freeze_method == 'pd_sc':
                percentile, pd_sc_step = freeze_args
                if self.handler.round == 0:
                    self.prev_handler_model_state_dict = copy.deepcopy(self.handler._model.state_dict())
                elif self.handler.round % pd_sc_step == 0:
                    # 开始进行更新name_list_to_update
                    if self.prev_handler_model_state_dict is not None:
                        result = pd_sc_to_update(percentile, self.prev_handler_model_state_dict, self.handler._model.state_dict(), name_list_to_update)
                        if len(result) == 0:
                            name_list_to_update = min_list_to_update
                        else:
                            name_list_to_update = result
                        wandbLogWrap({
                            "round" : self.handler.round,
                            "update_len" : len(name_list_to_update),
                        })
                    self.prev_handler_model_state_dict = copy.deepcopy(self.handler._model.state_dict())
            elif freeze_method == 'random_sync':
                num_freezable = freeze_args
                update_name_list_list = []
                for _ in range(len(sampled_clients)):
                    indices = random.sample(range(num_freezable), random.randint(2, num_freezable))
                    update_name_list_list.append(achieve_name_list_by_indices(total_name_list, indices))


            if freeze_method == 'static' or freeze_method == 'pd_sc':
                print(f"name list len: {len(name_list_to_update)}")
                # client side
                self.trainer.local_process_with_freeze(broadcast, sampled_clients, name_list_to_update)
            elif freeze_method == 'random_sync':
                print(update_name_list_list)
                self.trainer.local_process_with_freeze_2(broadcast, sampled_clients, update_name_list_list)
            self.trainer.scheduler.step()
            print("cur learning rate: ", self.trainer.scheduler.get_last_lr()[0])
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()
            # self.handler.evaluate()

    def evaluate(self):
        loss_, acc_ = self.handler.evaluate()


class StandalonePipelineWithCompression(object):
    def __init__(self, handler: ServerHandler, trainer: SerialClientTrainer):
        """Perform standalone simulation process.

        Args:
            handler (ServerHandler): _description_
            trainer (SerialClientTrainer): _description_
        """
        self.handler = handler
        self.trainer = trainer

        # initialization
        self.handler.num_clients = self.trainer.num_clients

        # 用于pd_sc
        self.prev_handler_model_state_dict = None

    def main(self, compression_args):
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package_2

            # client side
            total_bytes_communication = self.trainer.local_process_with_compression(broadcast, sampled_clients, compression_args)
            uploads = self.trainer.uplink_package

            # server side
            for pack in uploads:
                self.handler.load2(pack)

            # evaluate
            self.handler.evaluate_split()

            wandbLogWrap({
                "Round": self.handler.round,
                "total_bytes_communication": total_bytes_communication,
            })
            self.handler._LOGGER.info(
                "total bytes communication: {:.2f}".format(total_bytes_communication)
            )
            # self.handler.evaluate()

    def evaluate(self, bottom_model, top_model, test_loader, device, criterion):
        loss_ = AverageMeter()
        acc_ = AverageMeter()
        with torch.no_grad():
            bottom_model.eval()
            top_model.eval()
            for inputs, labels in test_loader:
                batch_size = len(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                output1 = bottom_model(inputs)
                outputs = top_model(output1)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                loss_.update(loss.item(), batch_size)
                acc_.update(torch.sum(predicted.eq(labels)).item() / batch_size, batch_size)

        return loss_.avg, acc_.avg
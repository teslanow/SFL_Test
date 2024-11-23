# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
import copy
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import Set

import math
import torch
from tqdm import tqdm
from ...core.client.trainer import ClientTrainer, SerialClientTrainer
from ...utils import Logger, SerializationTool

class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): :object of :class:`Logger`.
    """
    def __init__(self,
                 model:torch.nn.Module,
                 cuda:bool=False,
                 device:str=None,
                 logger:Logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


def model_set_freeze(model: torch.nn.Module, update_name_list: Set[str]):
    # 以下可能不保险
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name in update_name_list:
            param.requires_grad = True
            # print("need to update" , name)


class SGDSerialClientTrainer(SerialClientTrainer):
    """
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False, server_model=None) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []
        if server_model is not None:
            if cuda:
                self._server_model = deepcopy(server_model).cuda(device)
            else:
                self._server_model = deepcopy(server_model).cpu()
        else:
            self._server_model = None
    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr, step_size, gamma):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        if self._server_model is not None:
            self.server_optimizer = torch.optim.SGD(self._server_model.parameters(), lr)
            self.server_scheduler = torch.optim.lr_scheduler.StepLR(self.server_optimizer, step_size=5, gamma=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        """
        Returns: all_client_samples，表示每个client最终运行的samples的数目
        """
        model_parameters = payload[0]
        all_client_samples = {}
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            all_client_samples[id] = len(data_loader.dataset) * self.epochs
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)
        return all_client_samples



    def local_process_with_freeze(self, payload, id_list, update_name_list):
        """
        所有的client使用相同的freeze序列
        Args:
            payload:
            id_list:
            update_name_list: 一个list，list的每个element表示允许更新的tensor name
        """
        model_parameters = payload[0]
        all_client_samples = {}
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            all_client_samples[id] = len(data_loader.dataset) * self.epochs
            pack = self.train_with_freeze(model_parameters, data_loader, update_name_list)
            self.cache.append(pack)
        return all_client_samples

    def local_process_with_compression(self, payload, id_list, compression_args):
        bottom_model_parameters, top_model_parameters = payload[0], payload[1]
        total_bytes_communication = 0
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            # print(f"dataset len : {len(data_loader.dataset)}")
            pack, bytes_communication = self.train_with_compression(bottom_model_parameters, top_model_parameters, data_loader, compression_args)
            self.cache.append(pack)
            total_bytes_communication += bytes_communication
        return total_bytes_communication

    def local_process_with_freeze_2(self, payload, id_list, update_name_list_list):
        """
        client允许使用不同的freeze/update序列
        Args:
            update_name_list_list: 一个list，这个list的每个element也是一个list，该list和local_process_with_freeze中的update_name_list是一个意思
        """
        model_parameters = payload[0]
        index = 0
        all_client_samples = {}
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            update_name_list = update_name_list_list[index]
            index += 1
            pack = self.train_with_freeze(model_parameters, data_loader, update_name_list)
            all_client_samples[id] = len(data_loader.dataset) * self.epochs
            self.cache.append(pack)
        return all_client_samples

    def train(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]

    def train_with_freeze(self, model_parameters, train_loader, update_name_list):
        """Single round of local training for one client.

                Note:
                    Overwrite this method to customize the PyTorch training pipeline.

                Args:
                    update_name_list: a dict of parameter name to allow updating
                    model_parameters (torch.Tensor): serialized model parameters.
                    train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
                """

        self.set_model(model_parameters)
        # 冻住model
        model_set_freeze(self._model, update_name_list)
        self._model.train()
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]

    def train_with_compression(self, bottom_model_parameters, top_model_parameters, train_loader, compression_args):
        def compress_activation(x: torch.Tensor, compress_method: str, k_ratio=1.0, bits=32):
            """
            对x进行压缩，x需要被detach过
            Args:
                x: activation
                compress_method: normal，等

            Returns: compressed_x, theoretical bits to communication

            """
            with torch.no_grad():
                if compress_method == "normal":
                    return x, x.numel() * 4 * 8
                elif compress_method == "topk":
                    shape = x.shape
                    k = x.numel()
                    k = min(math.ceil(k * k_ratio), k)
                    flatten_tensor = x.view(-1)
                    abs_tensor = flatten_tensor.abs()
                    _, indices = torch.topk(abs_tensor, k)
                    mask = torch.zeros_like(flatten_tensor)
                    mask[indices] = 1
                    flatten_tensor = flatten_tensor * mask
                    flatten_tensor = flatten_tensor.reshape(shape)
                    return copy.deepcopy(flatten_tensor), k * 4 * 8
                elif compress_method == "quantize":
                    x_min = x.min()
                    x_max = x.max()
                    # 先量化
                    quant_min = 0
                    quant_max = (1 << bits) - 1
                    delta = (x_max - x_min) / (quant_max - quant_min)
                    zero_point = (- x_min / delta).round()
                    x_int = torch.round(x / delta)
                    x_quant = torch.clamp(x_int + zero_point, 0, quant_max)
                    x = ((x_quant - zero_point) * delta).float()
                    return copy.deepcopy(x), x.numel() * bits
                else:
                    raise NotImplementedError

        self.set_model(bottom_model_parameters)
        # 设置server model
        SerializationTool.deserialize_model(self._server_model, top_model_parameters)
        self._model.train()
        self._server_model.train()
        bytes_communication = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                smashed_data = output.clone().detach()
                # 压缩
                smashed_data, data_bits = compress_activation(smashed_data, compression_args["compress_method"], compression_args["k_ratio"],
                                                              compression_args["bits"])
                smashed_data.requires_grad = True
                output1 = self._server_model(smashed_data)
                loss = self.criterion(output1, target)

                self.server_optimizer.zero_grad()
                loss.backward()
                self.server_optimizer.step()
                # bottom_model的后向
                grad_smashed_data = smashed_data.grad.clone().detach()
                self.optimizer.zero_grad()
                output.backward(grad_smashed_data)
                self.optimizer.step()

                # 统计通信
                bytes_communication += (data_bits / 8)
            self.scheduler.step()
            self.server_scheduler.step()
        # 返回model parameter, [bottom_parameters, top_parameters]
        return [self.model_parameters, SerializationTool.serialize_model(self._server_model)], bytes_communication

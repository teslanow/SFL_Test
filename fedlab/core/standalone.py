# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
import copy
import heapq
import os
import pickle
import random
from typing import List
from collections import OrderedDict
import math
import numpy as np
import torch
from fedlab.utils.WandbWrapper import wandbLogWrap
from fedlab.utils.simulate_time import add_cur_time
import fedlab.utils.System_conf
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
from ..utils.ClientProperties import ClientPropertyManager
from ..utils.System_conf import get_cur_system_hetero
from ..utils.functional import AverageMeter

from test import *

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

    def set_clients_properties(self, client_profile_path, total_client_num):
        self.clientPropertyManager = ClientPropertyManager(client_profile_path, total_client_num)
        system_hetero = get_cur_system_hetero()
        if system_hetero == "compute_hetero":
            # all_comm = [v['communication']for v in self.clientPropertyManager.client_profiles]
            # self.common_communicate_bandwidth = sum(all_comm) / len(all_comm)
            self.common_communicate_bandwidth = self.clientPropertyManager.avg_comm
        elif system_hetero == "communicate_hetero":
            # all_comp = [v['computation']for v in self.clientPropertyManager.client_profiles]
            # self.common_compute_density = sum(all_comp) / len(all_comp)
            self.common_compute_density = self.clientPropertyManager.avg_comp
        elif system_hetero == "practical":
            pass
        else:
            raise NotImplementedError

    def main(self, save=False, model_density=False):
        all_sample_clients = []
        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            all_sample_clients.append(sampled_clients)
            broadcast = self.handler.downlink_package

            # client side
            all_client_sample_nums = self.trainer.local_process(broadcast, sampled_clients)

            uploads = self.trainer.uplink_package
            if all_client_sample_nums is not None:
                all_client_times = {}
                system_hetero = get_cur_system_hetero()
                for client_id in all_client_sample_nums.keys():
                    # 通信,Mb/s
                    if system_hetero == "compute_hetero":
                        compute_density = self.clientPropertyManager.get_client_profile(client_id)['computation']
                        communicate_bandwidth = self.common_communicate_bandwidth
                    elif system_hetero == "communicate_hetero":
                        compute_density = self.common_compute_density
                        communicate_bandwidth = self.clientPropertyManager.get_client_profile(client_id)['communication']
                    elif system_hetero == "practical":
                        compute_density = self.clientPropertyManager.get_client_profile(client_id)['computation']
                        communicate_bandwidth = self.clientPropertyManager.get_client_profile(client_id)['communication']
                    else:
                        raise NotImplementedError
                    param_size = 2 * broadcast[0].numel()
                    comm_time = param_size * 32 / 1000 / 1000 / communicate_bandwidth
                    run_time = model_density * all_client_sample_nums[client_id] / (compute_density * 1000000000)
                    # print(f"param num: {param_size}, comm time : {comm_time}, run time : {run_time}")
                    all_client_times[client_id] = comm_time + run_time
                time_this_round = max(list(all_client_times.values()))
                cur_time = add_cur_time(time_this_round)
                print(f'cur_time: {cur_time}')
            # server side
            for pack in uploads:
                self.handler.load(pack)

            # evaluate
            self.evaluate()
            # self.handler.evaluate()
            # 保存模型
            if save and self.handler.round % 1 == 0:
                path = "results/tmp_models/fedavg_pretrain/"
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(self.handler.model.state_dict(), os.path.join(path, f"{self.handler.round * self.trainer.epochs}.pt"))
            # 更新lr
            self.trainer.scheduler.step()
            print(self.trainer.optimizer.param_groups[0]['lr'])
        with open('results/all_sample_clients_fedavg.pkl', 'wb') as file:
            pickle.dump(all_sample_clients, file)
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


def get_runtimes_of_each_module(num_clients, speed_of_all_clients,
                                trainable_tensor_num, model: torch.nn.Module,
                                resnet34_per_module_flops, all_parameter_names, all_parameter_sizes):
    """

    Args:
        num_clients: clients数目
        speed_of_all_clients: 每个client的运行速度，单位为flop/s
        trainable_tensor_num: model的总的可trainable的参数数目
        model:
        resnet34_per_module_flops: 暂时这么硬编码
        all_parameter_names: 每个参数的名字，按照topological序
        all_parameter_sizes: 每个参数的size

    Returns:

    """
    runtimes = []
    for i in range(num_clients):
        c_runtimes = [[0.0 for _ in range(trainable_tensor_num)], [0.0 for _ in range(trainable_tensor_num)]]
        speed = speed_of_all_clients[i]
        for module_name, _ in model.named_modules():
            if module_name == '' or module_name not in resnet34_per_module_flops:
                continue
            flops_this_module = resnet34_per_module_flops[module_name]
            param_name_this_module = []
            param_size_this_module = []
            param_index_this_module = []
            for j, parameter_name in enumerate(all_parameter_names):
                if module_name in parameter_name:
                    param_name_this_module.append(parameter_name)
                    param_size_this_module.append(all_parameter_sizes[parameter_name])
                    param_index_this_module.append(j)
            param_ratio_this_module = np.array(param_size_this_module) / sum(param_size_this_module)
            for j, index in enumerate(param_index_this_module):
                c_runtimes[0][index] = param_ratio_this_module[j] * flops_this_module / speed
                c_runtimes[1][index] = param_ratio_this_module[j] * flops_this_module / speed
        runtimes.append(c_runtimes)
    runtimes = np.array(runtimes)
    return runtimes


class Circumstance:
    def __init__(self, run_times: np.ndarray, time_limit: np.ndarray,
                 all_gradients: List[List[torch.Tensor]], statistical_gradients: List[torch.Tensor],
                 alphas: List[float]):
        """
        初始化环境
        :param run_times:  一个n elements的array，array的每个element是两个个array，第一个表示[T_u0, T_u1, ...., T_uk]，第二个表示[T_c0, T_c1, ..., T_ck]
        其中T_ui表示计算第i个weight tensor的梯度所要的时间，T_ci表示计算activation梯度
        所要的时间
        :param time_limit: 一个n elements的array，每个element是这个client的时间上界
        :param all_gradients: 一个n element的list，每个element是一个k element的list，表示相应的tensor gradient
        :param statistical_gradients:一个k element的list，每个element是在全局数据集下的梯度tensor
        :param alphas: 一个n element的list
        """
        self.alphas = alphas
        self.statistical_gradients = statistical_gradients
        self.all_gradients = all_gradients
        self.run_times = run_times
        self.time_limit = time_limit
        self.penalty_fac = 100

    def cal_time(self, c_id, c_indicator, k):
        """

        :param c_id: client id
        :param c_indicator: client的k个tensor是否freeze的bit序列
        :param k:
        :return:
        """
        another_indicator = transfer_indicator(c_indicator)
        update_times, trans_times = self.run_times[c_id]
        assert len(update_times) == k
        assert len(trans_times) == k
        back_time = np.dot(c_indicator, update_times) + np.dot(another_indicator, trans_times)
        return back_time

    def cal_norm_of_k_th_tensor(self, k_id, k_indicator):
        """

        :param k_id: tensor id
        :param k_indicator: n个client对应的这个tensor是否冻结的bit序列
        :return:
        """
        stat_gradient = self.statistical_gradients[k_id]
        s = torch.zeros_like(self.statistical_gradients[k_id])
        for c_i, bit in enumerate(k_indicator):
            alpha = self.alphas[c_i]
            if bit == 0:
                s += (-alpha * stat_gradient)
            elif bit == 1:
                s += (alpha * (self.all_gradients[c_i][k_id] - stat_gradient))
            else:
                raise NotImplementedError
        return torch.norm(s, 2).item() ** 2

    def cal_fitness(self, bit_list, n, k):
        """
        根据bit_list计算fitness，先使用罚函数法
        :param bit_list: 一个list包含n个subpart，每个subpart包含k个bit，第i个bit=1表示client会更新tensor i
        :return: 带有罚项的fitness，越大越好
        """

        assert len(bit_list) == n * k
        all_back_times = []
        # 计算时间
        for c_id in range(n):
            indicator = bit_list[c_id * k: (c_id + 1) * k]
            back_time = self.cal_time(c_id, indicator, k)
            all_back_times.append(back_time)
        # 计算目标函数
        obj = 0
        for k_id in range(k):
            k_indicator = [bit_list[c_i * k + k_id] for c_i in range(n)]
            t = self.cal_norm_of_k_th_tensor(k_id, k_indicator)
            obj += t
        # 综合考虑obj
        # 罚函数法
        p = 0
        for c_id in range(n):
            limit = self.time_limit[c_id]
            b_time = all_back_times[c_id]
            p += (math.exp(max(0, b_time - limit) / limit) - 1)
        p = self.penalty_fac * p + 1
        return -obj * p


class Individual:
    def __init__(self, chromosome: np.ndarray, circum, n, k, ):
        """
        初始化个体
        :param chromosome: 对于n个clients，每个client有k个可冻结的tensor，则chromosome是一个长为nk的0-1序列；
        TODO：当chromosome格式变化的时候，update_fitness, mutation都需要变化
        """
        self.chromosome = chromosome
        self.n = n
        self.k = k
        self.fitness = self._update_fitness(circum)


    def _update_fitness(self, circum: Circumstance):
        def get_bit_list_1(ch):
            # bit_list: 一个list包含n个subpart，每个subpart包含k个bit，第i个bit=1表示client会更新tensor i
            return ch
        bit_list = get_bit_list_1(self.chromosome)
        return circum.cal_fitness(bit_list, self.n, self.k)

    def mutation(self, p):
        """
        变异
        :param p: 变异的位点数上界
        :return:
        """
        # 方法一：随机取n个位点
        for _ in range(p):
            i = random.randint(0, len(self.chromosome) - 1)
            self.chromosome[i] = 1 - self.chromosome[i]


def transfer_indicator(indicator: np.ndarray):
    re = np.ones_like(indicator)
    for i, bit in enumerate(indicator):
        if bit == 1:
            break
        re[i] = 0
    return re

def crossover(p1: Individual, p2: Individual, circum: Circumstance):
    # 单点交叉法
    assert len(p1.chromosome) == len(p2.chromosome)
    point = random.randint(1, len(p1.chromosome) - 1)
    ch = p1.chromosome.copy()
    # 可能时间/空间开销会很大
    if random.random() <= 0.5:
        ch[point:] = p2.chromosome[point:]
        # ch = p1.chromosome[:point] + p2.chromosome[point:]
    else:
        ch[:point] = p2.chromosome[:point]
        # ch = p2.chromosome[:point] + p1.chromosome[point:]
    children = Individual(ch, circum, p1.n, p1.k)
    return children

def decode_chromosome(chromosome: np.ndarray, all_parameter_names, n, k):
    update_name_list_list = []
    for c_id in range(n):
        update_name_list = []
        indicator = chromosome[c_id * k: (c_id + 1) * k]
        for i, t in enumerate(indicator):
            if t == 1:
                update_name_list.append(all_parameter_names[i])
        update_name_list_list.append(update_name_list)
    return update_name_list_list


class GA:
    def __init__(self, circum, n, k):
        self.k = k
        self.n = n
        self.circum = circum

    def run(self):
        epoch = 100
        population_size = 100
        mutation_prob = 0.4
        mutation_n_point = 5
        # 精英比例
        niubi_ratio = 0.1
        population = []
        circum, n, k = self.circum, self.n, self.k
        for _ in range(population_size):
            chromosome = np.array([random.choice([0, 1]) for _ in range(n * k)])
            individual = Individual(chromosome, circum, n, k)
            population.append(individual)
        # result = []
        # for _ in tqdm(range(epoch)):
        re_ch: Individual = None
        for _ in range(epoch):
            # 保留精英种子
            niubi_num = min(math.ceil(population_size * niubi_ratio), population_size)
            new_population = heapq.nlargest(niubi_num, population, key=lambda x: x.fitness)
            left_num = population_size - niubi_num
            for _ in range(left_num):
                p1 = random.choice(population)
                p2 = random.choice(population)
                ch = crossover(p1, p2, circum)
                if random.random() <= mutation_prob:
                    ch.mutation(mutation_n_point)
                new_population.append(ch)
            most_niubi_ch = max(population, key=lambda x: x.fitness)
            if re_ch is None or re_ch.fitness < most_niubi_ch.fitness:
                re_ch = copy.deepcopy(most_niubi_ch)
            # result.append(most_niubi_ch.fitness)
            # print(most_niubi_ch.chromosome)
            print(most_niubi_ch.fitness)
            population = new_population
        return re_ch.chromosome


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

    def set_clients_properties(self, client_profile_path, total_client_num):
        self.clientPropertyManager = ClientPropertyManager(client_profile_path, total_client_num)
        system_hetero = get_cur_system_hetero()
        if system_hetero == "compute_hetero":
            # all_comm = [v['communication']for v in self.clientPropertyManager.client_profiles]
            # self.common_communicate_bandwidth = sum(all_comm) / len(all_comm)
            self.common_communicate_bandwidth = self.clientPropertyManager.avg_comm
        elif system_hetero == "communicate_hetero":
            # all_comp = [v['computation']for v in self.clientPropertyManager.client_profiles]
            # self.common_compute_density = sum(all_comp) / len(all_comp)
            self.common_compute_density = self.clientPropertyManager.avg_comp
        elif system_hetero == "practical":
            pass
        else:
            raise NotImplementedError

    def main(self, freeze_args, freeze_method, forward_model_density=0):
        name_list_to_update = [str(name) for name, param in self.trainer.model.named_parameters()]
        total_name_list = copy.deepcopy(name_list_to_update)
        min_list_to_update = copy.deepcopy(name_list_to_update[-2:])

        all_parameter_names = []
        for name, parameter in self.handler.model.named_parameters():
            all_parameter_names.append(name)
        trainable_tensor_num = len(all_parameter_names)
        with open(os.path.join("/home/zhongxiangwei/search_projects/SFL_Test2/results/tmp_models/fedavg_pretrain", "resnet34-per-module-flops.pt"), 'rb') as f:
            resnet34_per_module_flops = pickle.load(f)
        all_parameter_sizes = {}
        for name, param in self.handler.model.named_parameters():
            all_parameter_sizes[name] = param.numel()

        while self.handler.if_stop is False:
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            num_clients = len(sampled_clients)
            speed_of_all_clients = self.clientPropertyManager.get_random_profile(num_clients)
            # 所有clients的run_speed
            run_speeds = [s['computation'] * 1000000000 for s in speed_of_all_clients]
            # 根据run_speeds计算每个client的每个算子的执行时间（只用run_flops除以计算速度）
            run_times = get_runtimes_of_each_module(num_clients, run_speeds, trainable_tensor_num,self.handler.model,
                                                   resnet34_per_module_flops, all_parameter_names, all_parameter_sizes)
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
            elif freeze_method == 'full':
                name_list_to_update = []
                for name, param in self.trainer.model.named_parameters():
                    name_list_to_update.append(name)
            elif freeze_method == 'selective':
                pass
            else:
                raise NotImplementedError

            if freeze_method == 'static' or freeze_method == 'pd_sc' or freeze_method == 'full':
                print(f"name list len: {len(name_list_to_update)}")
                # client side
                all_client_sample_nums = self.trainer.local_process_with_freeze(broadcast, sampled_clients, name_list_to_update)
                # 计算时间
                if all_client_sample_nums is not None:
                    all_client_times = {}
                    for i, client_id in enumerate(all_client_sample_nums.keys()):
                        # 暂时忽略通信
                        c_indicator = np.array([1 if name in name_list_to_update else 0 for name in all_parameter_names])
                        another_indicator = transfer_indicator(c_indicator)
                        update_times, trans_times = run_times[i]
                        back_time = np.dot(c_indicator, update_times) + np.dot(another_indicator, trans_times)
                        run_t = back_time + forward_model_density / run_speeds[i]
                        all_client_times[client_id] = run_t * all_client_sample_nums[client_id]
                    time_this_round = max(list(all_client_times.values()))
                    cur_time = add_cur_time(time_this_round)
                    print(f'cur_time: {cur_time}')
            elif freeze_method == 'random_sync':
                # print(update_name_list_list)
                all_client_sample_nums = self.trainer.local_process_with_freeze_2(broadcast, sampled_clients, update_name_list_list)
                if all_client_sample_nums is not None:
                    all_client_times = {}
                    for i, client_id in enumerate(all_client_sample_nums.keys()):
                        # 暂时忽略通信
                        name_list_to_update = update_name_list_list[i]
                        c_indicator = np.array([1 if name in name_list_to_update else 0 for name in all_parameter_names])
                        another_indicator = transfer_indicator(c_indicator)
                        update_times, trans_times = run_times[i]
                        back_time = np.dot(c_indicator, update_times) + np.dot(another_indicator, trans_times)
                        run_t = back_time + forward_model_density / run_speeds[i]
                        all_client_times[client_id] = run_t * all_client_sample_nums[client_id]
                    time_this_round = max(list(all_client_times.values()))
                    cur_time = add_cur_time(time_this_round)
                    print(f'cur_time: {cur_time}')
            elif freeze_method == 'selective':
                # 构造circumstance
                device = 'cuda:6'
                overall_dataset = load_train_dataset_from_clients_total(range(100), "/data/zhongxiangwei/data/CIFAR10")
                overall_loader = DataLoader(overall_dataset, batch_size=256, shuffle=True)
                cri = torch.nn.CrossEntropyLoss()
                model = self.handler.model
                optim = torch.optim.SGD(model.parameters(), lr=0.001)
                avg_grads = statistical_gradient(model, overall_loader, optim, device, criterion=cri)
                with open('/home/zhongxiangwei/search_projects/SFL_Test2/results/tmp_models2/fedavg_pretrain/all.pkl', 'wb') as f:
                    pickle.dump(avg_grads, f)
                statistical_gradients = []
                for parameter_name in all_parameter_names:
                    assert parameter_name in avg_grads
                    statistical_gradients.append(avg_grads[parameter_name])
                all_gradients = []
                for client_id in sampled_clients:
                    print("train on {client_id}".format(client_id=client_id))
                    client_dataset = load_train_dataset_from_clients(client_id, "/data/zhongxiangwei/data/CIFAR10")
                    client_loader = DataLoader(client_dataset, batch_size=256, shuffle=True)
                    this_gradient = statistical_gradient(model, client_loader, optim, device, cri)
                    with open(f'/home/zhongxiangwei/search_projects/SFL_Test2/results/tmp_models2/fedavg_pretrain/client-{client_id}.pkl','wb') as f:
                        pickle.dump(this_gradient, f)
                    grads = []
                    for parameter_name in all_parameter_names:
                        assert parameter_name in this_gradient
                        grads.append(this_gradient[parameter_name])
                    all_gradients.append(grads)
                exit(-1)
                normal_times = [2 * forward_model_density / s for s in run_speeds]
                largest_time = max(normal_times)
                time_limit = np.array([largest_time * 0.5 for _ in range(num_clients)])
                alphas = [1 / num_clients for _ in range(num_clients)]
                circum = Circumstance(run_times, time_limit, all_gradients, statistical_gradients, alphas)
                ga = GA(circum, num_clients, trainable_tensor_num)
                chromosome = ga.run()
                update_name_list_list = decode_chromosome(chromosome, all_parameter_names, num_clients, trainable_tensor_num)
                all_client_sample_nums = self.trainer.local_process_with_freeze_2(broadcast, sampled_clients, update_name_list_list)
                if all_client_sample_nums is not None:
                    all_client_times = {}
                    for i, client_id in enumerate(all_client_sample_nums.keys()):
                        # 暂时忽略通信
                        name_list_to_update = update_name_list_list[i]
                        c_indicator = np.array([1 if name in name_list_to_update else 0 for name in all_parameter_names])
                        another_indicator = transfer_indicator(c_indicator)
                        update_times, trans_times = run_times[i]
                        back_time = np.dot(c_indicator, update_times) + np.dot(another_indicator, trans_times)
                        run_t = back_time + forward_model_density / run_speeds[i]
                        all_client_times[client_id] = run_t * all_client_sample_nums[client_id]
                    time_this_round = max(list(all_client_times.values()))
                    cur_time = add_cur_time(time_this_round)
                    print(f'cur_time: {cur_time}')
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
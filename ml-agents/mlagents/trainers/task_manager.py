from typing import Dict, List, Tuple, Optional
from mlagents.trainers.settings import (
    TaskParameterSettings,
    ParameterRandomizationSettings,
)
from collections import defaultdict
from mlagents.trainers.training_status import GlobalTrainingStatus, StatusType

from mlagents_envs.logging_util import get_logger

from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.active_learning import ActiveLearningTaskSampler, sample_random_points

logger = get_logger(__name__)

import torch
import numpy as np

class TaskManager:
    def __init__(
        self,
        settings: Optional[Dict[str, TaskParameterSettings]] = None,
        restore: bool = False,
    ):
        """
        EnvironmentParameterManager manages all the environment parameters of a training
        session. It determines when parameters should change and gives access to the
        current sampler of each parameter.
        :param settings: A dictionary from environment parameter to
        EnvironmentParameterSettings.
        :param restore: If true, the EnvironmentParameterManager will use the
        GlobalTrainingStatus to try and reload the lesson status of each environment
        parameter.
        """
        if settings is None:
            settings = {}
        self._dict_settings = settings
        
        self.behavior_names = list(self._dict_settings.keys())
        self.param_names = {name: list(self._dict_settings[name].parameters.keys()) for name in self.behavior_names}
        self._taskSamplers = {}
        self.report_buffer = []
        self.num_repeat = {name: 1 for name in self.behavior_names}
        self.task_completed = {name: defaultdict(list) for name in self.behavior_names}
        self.num_batch = {name: 1 for name in self.behavior_names}

        for behavior_name in self.behavior_names:
            lows = []
            highs = []
            parameters = self._dict_settings[behavior_name].parameters
            for parameter_name in self.param_names[behavior_name]:
                low = parameters[parameter_name].min_value
                high = parameters[parameter_name].max_value
                lows.append(low)
                highs.append(high)    
            task_ranges = torch.tensor([lows, highs]).float().T
            self.num_repeat[behavior_name] = self._dict_settings[behavior_name].num_repeat
            self.num_batch[behavior_name] = self._dict_settings[behavior_name].num_batch
            
            active_hyps = self._dict_settings[behavior_name].active_learning
            if active_hyps:
                self._taskSamplers[behavior_name] = ActiveLearningTaskSampler(task_ranges, 
                    warmup_steps=active_hyps.warmup_steps, capacity=active_hyps.capacity,
                    num_mc=active_hyps.num_mc, beta=active_hyps.beta,
                    raw_samples=active_hyps.raw_samples, num_restarts=active_hyps.num_restarts,
                )
            else:
                self._taskSamplers[behavior_name] = lambda n: sample_random_points(task_ranges.T, n)
        print("num batch", self.num_batch)
        self.t = {name: 0.0 for name in self.behavior_names}    
        self.counter = {name: 0 for name in self.behavior_names}    

    def _make_task(self, behavior_name, tau):
        task = {}
        for i, name in enumerate(self.param_names[behavior_name]):
            task[name] = tau[i]
        return task

    def _build_tau(self, behavior_name, task, time):
        tau = []
        for name in self.param_names[behavior_name]:
            tau.append(task[name])
        tau.append(time)
        return torch.tensor(tau).float()

    def get_tasks(self, behavior_name, num_samples) -> Dict[str, ParameterRandomizationSettings]:
        """
        TODO
        """
        behavior_name = [bname for bname in self.behavior_names if bname in behavior_name][0] # TODO make work with actual behavior names
        current_time = self.t[behavior_name] + 1

        if isinstance(self._taskSamplers[behavior_name], ActiveLearningTaskSampler):
            num_points = max(num_samples, self.num_batch[behavior_name])
            taus = self._taskSamplers[behavior_name].get_design_points(num_points=num_points, time=current_time).data.numpy().tolist()
        else:
            taus  = self._taskSamplers[behavior_name](num_samples).tolist()
        # print("sampled taus", current_time, taus)
        tasks = [self._make_task(behavior_name, tau) for tau in taus]
        self.report_buffer.extend(tasks)
        tasks_repeated = []
        for i in range(self.num_repeat[behavior_name]):
            tasks_repeated.extend(tasks)

        return tasks_repeated

    def add_run(self, behavior_name, tau, perf):
        k = tuple(tau.data.numpy().flatten()[:-1].tolist())
        self.task_completed[behavior_name][k].append(perf)

    def get_data(self, behavior_name, last=True):
        taus = []
        perfs = []
        t = self.t[behavior_name]
        for k, v in self.task_completed[behavior_name].items():
            tau = torch.tensor(k + (t,)).float()
            taus.append(tau)
            if last:
                perf = v[-1]
            else:
                perf = np.mean(v)
            perfs.append(perf)

        X = torch.stack(taus, dim=0)
        Y = torch.tensor(perfs).float().reshape(-1, 1)
        return X, Y
            

    def update(self, behavior_name: str, task_perfs: List[Tuple[Dict, float]]
    ) -> Tuple[bool, bool]:
        """
        TODO
        """

        must_reset = False
        updated = False
        behavior_name = [bname for bname in self.behavior_names if bname in behavior_name][0] # TODO make work with actual behavior names
        if isinstance(self._taskSamplers[behavior_name], ActiveLearningTaskSampler):
            for task, perf in task_perfs:
                # perfs.append(perf)
                # self.t[behavior_name] = self.t[behavior_name] + 1
                tau = self._build_tau(behavior_name, task, self.t[behavior_name])
                # taus.append(tau)
                self.add_run(behavior_name, tau, perf)
            
            N = len(task_perfs)
            self.counter[behavior_name] += N
            M = self.num_repeat[behavior_name] * self.num_batch[behavior_name]
            if self.counter[behavior_name] >= M:
                updated = True
                self.t[behavior_name] += 1
                X, Y = self.get_data(behavior_name, last=True)
                self.task_completed[behavior_name] = defaultdict(list)
                self._taskSamplers[behavior_name].update_model(X, Y, refit=True)
        
        return updated, must_reset


def uniform_sample(ranges, num_samples):
    low = ranges[:, 0]
    high = ranges[:, 1]
    points = np.random.uniform(low=low, high=high, size=num_samples).reshape(num_samples, -1)
    return points

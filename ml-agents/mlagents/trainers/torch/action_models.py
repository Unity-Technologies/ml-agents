import abc
from typing import List, Tuple
from mlagents.torch_utils import torch, nn
import numpy as np
import math
from mlagents.trainers.torch.layers import linear_layer, Initialization
from mlagents.trainers.torch.distributions import DistInstance, DiscreteDistInstance, GaussianDistribution, MultiCategoricalDistribution

from mlagents.trainers.torch.utils import ModelUtils

EPSILON = 1e-7  # Small value to avoid divide by zero




class ActionModel(nn.Module, abc.ABC):
    #@abc.abstractmethod
    #def entropy(self, action_list: np.ndarray) -> torch.Tensor:
    #    pass
    #@abc.abstractmethod
    #def log_probs(self, action_list: np.ndarray) -> torch.Tensor:
    #    pass

    def _sample_action(self, dists: List[DistInstance]) -> List[torch.Tensor]:
        actions = []
        for action_dist in dists:
            action = action_dist.sample()
            actions.append(action)
        return actions

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        pass

class HybridActionModel(ActionModel):
    def __init__(
        self,
        hidden_size: int,
        continuous_act_size: int,
        discrete_act_size: List[int],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.encoding_size = hidden_size
        self.continuous_act_size = continuous_act_size
        self.discrete_act_size = discrete_act_size
        self.continuous_distribution = None #: List[GaussianDistribution] = []
        self.discrete_distribution = None #: List[MultiCategoricalDistribution] = []
        if continuous_act_size > 0:
            self.continuous_distribution = GaussianDistribution(
                    self.encoding_size,
                    continuous_act_size,
                    conditional_sigma=conditional_sigma,
                    tanh_squash=tanh_squash,
                )
        if len(discrete_act_size) > 0:
            self.discrete_distribution = MultiCategoricalDistribution(self.encoding_size, discrete_act_size)
            


    def evaluate(self, inputs: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        continuous_dists, discrete_dists = self._get_dists(inputs, masks)
        continuous_actions, discrete_actions= torch.split(actions, self.continuous_act_size, dim=1) 

        continuous_action_list = [continuous_actions[..., i] for i in range(continuous_actions.shape[-1])]
        continuous_log_probs, continuous_entropies, _ = ModelUtils.get_probs_and_entropy(continuous_action_list, continuous_dists)

        discrete_action_list = [discrete_actions[..., i] for i in range(discrete_actions.shape[-1])]
        discrete_log_probs, discrete_entropies, _ = ModelUtils.get_probs_and_entropy(discrete_action_list, discrete_dists)

        log_probs = torch.add(continuous_log_probs, discrete_log_probs) 
        entropies = torch.add(continuous_entropies, discrete_entropies) 
        return log_probs, entropies

    def get_action_out(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        continuous_dists, discrete_dists = self._get_dists(inputs, masks)
        dists = continuous_dists + discrete_dists
        return torch.cat([dist.exported_model_output() for dist in dists], dim=1)

    def _get_dists(self, inputs: torch.Tensor, masks: torch.Tensor) -> Tuple[List[DistInstance], List[DiscreteDistInstance]]:
        #continuous_distributions: List[DistInstance] = []
        #discrete_distributions: List[DiscreteDistInstance] = []
        continuous_dist_instances = self.continuous_distribution(inputs)# for continuous_dist in self.continuous_distributions]
        discrete_dist_instances = self.discrete_distribution(inputs, masks)# for discrete_dist in self.discrete_distributions]
        #for continuous_dist in self.continuous_distributions:
        #    continuous_distributions += continuous_dist(inputs)
        #for discrete_dist in self.discrete_distributions:
        #    discrete_distributions += discrete_dist(inputs, masks)
        return continuous_dist_instances, discrete_dist_instances

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_dists, discrete_dists = self._get_dists(inputs, masks)

        continuous_action_list = self._sample_action(continuous_dists)
        continuous_entropies, continuous_log_probs, continuous_all_probs = ModelUtils.get_probs_and_entropy(
            continuous_action_list, continuous_dists
        )
        continuous_actions = torch.stack(continuous_action_list, dim=-1)
        continuous_actions = continuous_actions[:, :, 0]
        

        discrete_action_list = self._sample_action(discrete_dists)
        discrete_entropies, discrete_log_probs, discrete_all_probs = ModelUtils.get_probs_and_entropy(
            discrete_action_list, discrete_dists
        )
        discrete_actions = torch.stack(discrete_action_list, dim=-1)
        discrete_actions = discrete_actions[:, 0, :]

        action = torch.cat([continuous_actions, discrete_actions.type(torch.float)], axis=1) 
        log_probs = torch.add(continuous_log_probs, discrete_log_probs) 
        entropies = torch.add(continuous_entropies, discrete_entropies) 

        #print("ac",action)
        #print("clp",continuous_log_probs)
        #print("dlp",discrete_log_probs)
        #print("lp",log_probs)
        #print("en",entropies)
        return (action, log_probs, entropies)

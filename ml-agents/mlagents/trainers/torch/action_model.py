from typing import List, Tuple
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch.distributions import (
    DistInstance,
    GaussianDistribution,
    MultiCategoricalDistribution,
)

from mlagents.trainers.torch.utils import ModelUtils, AgentAction, ActionLogProbs
from mlagents_envs.base_env import ActionSpec

EPSILON = 1e-7  # Small value to avoid divide by zero


class ActionModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.encoding_size = hidden_size
        self.action_spec = action_spec
        self._distributions = torch.nn.ModuleList()

        if self.action_spec.continuous_size > 0:
            self._distributions.append(
                GaussianDistribution(
                    self.encoding_size,
                    self.action_spec.continuous_size,
                    conditional_sigma=conditional_sigma,
                    tanh_squash=tanh_squash,
                )
            )

        if self.action_spec.discrete_size > 0:
            self._distributions.append(
                MultiCategoricalDistribution(
                    self.encoding_size, self.action_spec.discrete_branches
                )
            )

    def _sample_action(self, dists: List[DistInstance]) -> List[torch.Tensor]:
        """
        Samples actions from list of distribution instances
        """
        actions = []
        for action_dist in dists:
            action = action_dist.sample()
            actions.append(action)
        return actions

    def _get_dists(
        self, inputs: torch.Tensor, masks: torch.Tensor
    ) -> List[DistInstance]:
        distribution_instances: List[DistInstance] = []
        for distribution in self._distributions:
            dist_instances = distribution(inputs, masks)
            for dist_instance in dist_instances:
                distribution_instances.append(dist_instance)
        return distribution_instances

    def evaluate(
        self, inputs: torch.Tensor, masks: torch.Tensor, actions: AgentAction
    ) -> Tuple[ActionLogProbs, torch.Tensor]:
        dists = self._get_dists(inputs, masks)
        action_list = actions.to_tensor_list()
        log_probs_list, entropies, _ = ModelUtils.get_probs_and_entropy(
            action_list, dists
        )
        log_probs = ActionLogProbs.create(log_probs_list, self.action_spec)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)
        return log_probs, entropy_sum

    def get_action_out(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        dists = self._get_dists(inputs, masks)
        return torch.cat([dist.exported_model_output() for dist in dists], dim=1)

    def forward(
        self, inputs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        dists = self._get_dists(inputs, masks)
        action_list = self._sample_action(dists)
        log_probs_list, entropies, all_logs_list = ModelUtils.get_probs_and_entropy(
            action_list, dists
        )
        actions = AgentAction.create(action_list, self.action_spec)
        log_probs = ActionLogProbs.create(
            log_probs_list, self.action_spec, all_logs_list
        )
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)
        return (actions, log_probs, entropy_sum)

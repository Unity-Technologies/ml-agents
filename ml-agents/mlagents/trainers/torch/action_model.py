from typing import List, Tuple, NamedTuple, Optional
from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch.distributions import (
    DistInstance,
    DiscreteDistInstance,
    GaussianDistribution,
    MultiCategoricalDistribution,
)

from mlagents.trainers.torch.utils import AgentAction, ActionLogProbs
from mlagents_envs.base_env import ActionSpec

EPSILON = 1e-7  # Small value to avoid divide by zero


class DistInstances(NamedTuple):
    continuous: DistInstance
    discrete: List[DiscreteDistInstance]


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
        self._continuous_distribution = None
        self._discrete_distribution = None

        if self.action_spec.continuous_size > 0:
            self._continuous_distribution = GaussianDistribution(
                self.encoding_size,
                self.action_spec.continuous_size,
                conditional_sigma=conditional_sigma,
                tanh_squash=tanh_squash,
            )

        if self.action_spec.discrete_size > 0:
            self._discrete_distribution = MultiCategoricalDistribution(
                self.encoding_size, self.action_spec.discrete_branches
            )

    def _sample_action(self, dists: DistInstances) -> AgentAction:
        """
        Samples actions from a DistInstances tuple
        """
        continuous_action: Optional[torch.Tensor] = None
        discrete_action: Optional[List[torch.Tensor]] = None
        if self.action_spec.continuous_size > 0:
            continuous_action = dists.continuous.sample()
        if self.action_spec.discrete_size > 0:
            discrete_action = []
            for discrete_dist in dists.discrete:
                discrete_action.append(discrete_dist.sample())
        return AgentAction(continuous_action, discrete_action)

    def _get_dists(self, inputs: torch.Tensor, masks: torch.Tensor) -> DistInstances:
        continuous_dist: Optional[DistInstance] = None
        discrete_dist: Optional[List[DiscreteDistInstance]] = None
        if self.action_spec.continuous_size > 0:
            continuous_dist = self._continuous_distribution(inputs, masks)
        if self.action_spec.discrete_size > 0:
            discrete_dist = self._discrete_distribution(inputs, masks)
        return DistInstances(continuous_dist, discrete_dist)

    def _get_probs_and_entropy(
        self, actions: AgentAction, dists: DistInstances
    ) -> Tuple[ActionLogProbs, torch.Tensor]:

        entropies_list: List[torch.Tensor] = []
        continuous_log_prob: Optional[torch.Tensor] = None
        discrete_log_probs: Optional[List[torch.Tensor]] = None
        all_discrete_log_probs: Optional[List[torch.Tensor]] = None
        if self.action_spec.continuous_size > 0:
            continuous_log_prob = dists.continuous.log_prob(actions.continuous_tensor)
            entropies_list.append(dists.continuous.entropy())
        if self.action_spec.discrete_size > 0:
            discrete_log_probs = []
            all_discrete_log_probs = []
            for discrete_action, discrete_dist in zip(
                actions.discrete_list, dists.discrete
            ):
                discrete_log_prob = discrete_dist.log_prob(discrete_action)
                entropies_list.append(discrete_dist.entropy())
                discrete_log_probs.append(discrete_log_prob)
                all_discrete_log_probs.append(discrete_dist.all_log_prob())
        action_log_probs = ActionLogProbs(
            continuous_log_prob, discrete_log_probs, all_discrete_log_probs
        )
        entropies = torch.cat(entropies_list, dim=1)
        return action_log_probs, entropies

    def evaluate(
        self, inputs: torch.Tensor, masks: torch.Tensor, actions: AgentAction
    ) -> Tuple[ActionLogProbs, torch.Tensor]:
        dists = self._get_dists(inputs, masks)
        log_probs, entropies = self._get_probs_and_entropy(actions, dists)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)
        return log_probs, entropy_sum

    def get_action_out(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        dists = self._get_dists(inputs, masks)
        out_list: List[torch.Tensor] = []
        if self.action_spec.continuous_size > 0:
            out_list.append(dists.continuous.exported_model_output())
        if self.action_spec.discrete_size > 0:
            for discrete_dist in dists.discrete:
                out_list.append(discrete_dist.exported_model_output())
        return torch.cat(out_list, dim=1)

    def forward(
        self, inputs: torch.Tensor, masks: torch.Tensor
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor]:
        dists = self._get_dists(inputs, masks)
        actions = self._sample_action(dists)
        log_probs, entropies = self._get_probs_and_entropy(actions, dists)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)
        return (actions, log_probs, entropy_sum)

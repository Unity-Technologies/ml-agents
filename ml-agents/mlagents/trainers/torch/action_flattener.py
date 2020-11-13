from typing import List
from mlagents.torch_utils import torch

from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.utils import ModelUtils


class ActionFlattener:
    def __init__(self, action_spec: ActionSpec):
        self._specs = action_spec

    @property
    def flattened_size(self) -> int:
        return self._specs.continuous_size + sum(self._specs.discrete_branches)

    def forward(self, action: AgentAction) -> torch.Tensor:
        action_list: List[torch.Tensor] = []
        if self._specs.continuous_size > 0:
            action_list.append(action.continuous_tensor)
        if self._specs.discrete_size > 0:
            flat_discrete = torch.cat(
                ModelUtils.actions_to_onehot(
                    torch.as_tensor(action.discrete_tensor, dtype=torch.long),
                    self._specs.discrete_branches,
                ),
                dim=1,
            )
            action_list.append(flat_discrete)
        return torch.cat(action_list, dim=1)

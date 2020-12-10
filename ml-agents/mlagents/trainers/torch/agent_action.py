from typing import List, Optional, NamedTuple, Dict
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.utils import ModelUtils
from mlagents_envs.base_env import ActionTuple


class AgentAction(NamedTuple):
    """
    A NamedTuple containing the tensor for continuous actions and list of tensors for
    discrete actions. Utility functions provide numpy <=> tensor conversions to be
    sent as actions to the environment manager as well as used by the optimizers.
    :param continuous_tensor: Torch tensor corresponding to continuous actions
    :param discrete_list: List of Torch tensors each corresponding to discrete actions
    """

    continuous_tensor: torch.Tensor
    discrete_list: Optional[List[torch.Tensor]]

    @property
    def discrete_tensor(self):
        """
        Returns the discrete action list as a stacked tensor
        """
        return torch.stack(self.discrete_list, dim=-1)

    def to_action_tuple(self, clip: bool = False) -> ActionTuple:
        """
        Returns an ActionTuple
        """
        action_tuple = ActionTuple()
        if self.continuous_tensor is not None:
            _continuous_tensor = self.continuous_tensor
            if clip:
                _continuous_tensor = torch.clamp(_continuous_tensor, -3, 3) / 3
            continuous = ModelUtils.to_numpy(_continuous_tensor)
            action_tuple.add_continuous(continuous)
        if self.discrete_list is not None:
            discrete = ModelUtils.to_numpy(self.discrete_tensor[:, 0, :])
            action_tuple.add_discrete(discrete)
        return action_tuple

    @staticmethod
    def from_dict(buff: Dict[str, np.ndarray]) -> "AgentAction":
        """
        A static method that accesses continuous and discrete action fields in an AgentBuffer
        and constructs the corresponding AgentAction from the retrieved np arrays.
        """
        continuous: torch.Tensor = None
        discrete: List[torch.Tensor] = None  # type: ignore
        if "continuous_action" in buff:
            continuous = ModelUtils.list_to_tensor(buff["continuous_action"])
        if "discrete_action" in buff:
            discrete_tensor = ModelUtils.list_to_tensor(
                buff["discrete_action"], dtype=torch.long
            )
            discrete = [
                discrete_tensor[..., i] for i in range(discrete_tensor.shape[-1])
            ]
        return AgentAction(continuous, discrete)

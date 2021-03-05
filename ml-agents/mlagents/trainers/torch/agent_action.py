from typing import List, Optional, NamedTuple
import itertools
import numpy as np
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey
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
    def discrete_tensor(self) -> torch.Tensor:
        """
        Returns the discrete action list as a stacked tensor
        """
        if self.discrete_list is not None and len(self.discrete_list) > 0:
            return torch.stack(self.discrete_list, dim=-1)
        else:
            return torch.empty(0)

    def slice(self, start: int, end: int) -> "AgentAction":
        """
        Returns an AgentAction with the continuous and discrete tensors slices
        from index start to index end.
        """
        _cont = None
        _disc_list = []
        if self.continuous_tensor is not None:
            _cont = self.continuous_tensor[start:end]
        if self.discrete_list is not None and len(self.discrete_list) > 0:
            for _disc in self.discrete_list:
                _disc_list.append(_disc[start:end])
        return AgentAction(_cont, _disc_list)

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
    def from_buffer(buff: AgentBuffer) -> "AgentAction":
        """
        A static method that accesses continuous and discrete action fields in an AgentBuffer
        and constructs the corresponding AgentAction from the retrieved np arrays.
        """
        continuous: torch.Tensor = None
        discrete: List[torch.Tensor] = None  # type: ignore
        if BufferKey.CONTINUOUS_ACTION in buff:
            continuous = ModelUtils.list_to_tensor(buff[BufferKey.CONTINUOUS_ACTION])
        if BufferKey.DISCRETE_ACTION in buff:
            discrete_tensor = ModelUtils.list_to_tensor(
                buff[BufferKey.DISCRETE_ACTION], dtype=torch.long
            )
            discrete = [
                discrete_tensor[..., i] for i in range(discrete_tensor.shape[-1])
            ]
        return AgentAction(continuous, discrete)

    @staticmethod
    def _group_agent_action_from_buffer(
        buff: AgentBuffer, cont_action_key: BufferKey, disc_action_key: BufferKey
    ) -> List["AgentAction"]:
        """
        Extracts continuous and discrete groupmate actions, as specified by BufferKey, and
        returns a List of AgentActions that correspond to the groupmate's actions. List will
        be of length equal to the maximum number of groupmates in the buffer. Any spots where
        there are less agents than maximum, the actions will be padded with 0's.
        """
        continuous_tensors: List[torch.Tensor] = []
        discrete_tensors: List[torch.Tensor] = []
        if cont_action_key in buff:
            padded_batch = buff[cont_action_key].padded_to_batch()
            continuous_tensors = [
                ModelUtils.list_to_tensor(arr) for arr in padded_batch
            ]
        if disc_action_key in buff:
            padded_batch = buff[disc_action_key].padded_to_batch(dtype=np.long)
            discrete_tensors = [
                ModelUtils.list_to_tensor(arr, dtype=torch.long) for arr in padded_batch
            ]

        actions_list = []
        for _cont, _disc in itertools.zip_longest(
            continuous_tensors, discrete_tensors, fillvalue=None
        ):
            if _disc is not None:
                _disc = [_disc[..., i] for i in range(_disc.shape[-1])]
            actions_list.append(AgentAction(_cont, _disc))
        return actions_list

    @staticmethod
    def group_from_buffer(buff: AgentBuffer) -> List["AgentAction"]:
        """
        A static method that accesses next group continuous and discrete action fields in an AgentBuffer
        and constructs a padded List of AgentActions that represent the group agent actions.
        The List is of length equal to max number of groupmate agents in the buffer, and the AgentBuffer iss
        of the same length as the buffer. Empty spots (e.g. when agents die) are padded with 0.
        :param buff: AgentBuffer of a batch or trajectory
        :return: List of groupmate's AgentActions
        """
        return AgentAction._group_agent_action_from_buffer(
            buff, BufferKey.GROUP_CONTINUOUS_ACTION, BufferKey.GROUP_DISCRETE_ACTION
        )

    @staticmethod
    def group_from_buffer_next(buff: AgentBuffer) -> List["AgentAction"]:
        """
        A static method that accesses next group continuous and discrete action fields in an AgentBuffer
        and constructs a padded List of AgentActions that represent the next group agent actions.
        The List is of length equal to max number of groupmate agents in the buffer, and the AgentBuffer iss
        of the same length as the buffer. Empty spots (e.g. when agents die) are padded with 0.
        :param buff: AgentBuffer of a batch or trajectory
        :return: List of groupmate's AgentActions
        """
        return AgentAction._group_agent_action_from_buffer(
            buff, BufferKey.GROUP_NEXT_CONT_ACTION, BufferKey.GROUP_NEXT_DISC_ACTION
        )

    def to_flat(self, discrete_branches: List[int]) -> torch.Tensor:
        """
        Flatten this AgentAction into a single torch Tensor of dimension (batch, num_continuous + num_one_hot_discrete).
        Discrete actions are converted into one-hot and concatenated with continuous actions.
        :param discrete_branches: List of sizes for discrete actions.
        :return: Tensor of flattened actions.
        """
        # if there are any discrete actions, create one-hot
        if self.discrete_list is not None and len(self.discrete_list) > 0:
            discrete_oh = ModelUtils.actions_to_onehot(
                self.discrete_tensor, discrete_branches
            )
            discrete_oh = torch.cat(discrete_oh, dim=1)
        else:
            discrete_oh = torch.empty(0)
        return torch.cat([self.continuous_tensor, discrete_oh], dim=-1)

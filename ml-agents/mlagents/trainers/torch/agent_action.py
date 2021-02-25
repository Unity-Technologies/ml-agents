from typing import List, Optional, NamedTuple
import itertools
import numpy as np
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey, AgentBufferField
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
    def _padded_time_to_batch(
        agent_buffer_field: AgentBufferField, dtype: torch.dtype = torch.float32
    ) -> List[torch.Tensor]:
        """
        Pad actions and convert to tensor. Pad the data with 0s where there is no
        data. 0 is used instead of NaN because NaN is not a valid entry for integer
        tensors, as used for discrete actions.
        """
        action_shape = None
        for _action in agent_buffer_field:
            # _action could be an empty list if there are no group agents in this
            # step. Find the first non-empty list and use that shape.
            if _action:
                action_shape = _action[0].shape
                break
        # If there were no groupmate agents in the entire batch, return an empty List.
        if action_shape is None:
            return []

        # Convert to tensor while padding with 0's
        new_list = list(
            map(
                lambda x: ModelUtils.list_to_tensor(x, dtype=dtype),
                itertools.zip_longest(
                    *agent_buffer_field, fillvalue=np.full(action_shape, 0)
                ),
            )
        )
        return new_list

    @staticmethod
    def _group_from_buffer(
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
            continuous_tensors = AgentAction._padded_time_to_batch(
                buff[cont_action_key]
            )
        if disc_action_key in buff:
            discrete_tensors = AgentAction._padded_time_to_batch(
                buff[disc_action_key], dtype=torch.long
            )

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
        return AgentAction._group_from_buffer(
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
        return AgentAction._group_from_buffer(
            buff, BufferKey.GROUP_NEXT_CONT_ACTION, BufferKey.GROUP_NEXT_DISC_ACTION
        )

    def to_flat(self, discrete_branches: List[int]) -> torch.Tensor:
        """
        Flatten this AgentAction into a single torch Tensor of dimension (batch, num_continuous + num_one_hot_discrete).
        Discrete actions are converted into one-hot and concatenated with continuous actions.
        :param discrete_branches: List of sizes for discrete actions.
        :return: Tensor of flattened actions.
        """
        discrete_oh = ModelUtils.actions_to_onehot(
            self.discrete_tensor, discrete_branches
        )
        discrete_oh = torch.cat(discrete_oh, dim=1)
        return torch.cat([self.continuous_tensor, discrete_oh], dim=-1)

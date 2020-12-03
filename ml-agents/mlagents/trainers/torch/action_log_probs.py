from typing import List, Optional, NamedTuple, Dict
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.utils import ModelUtils
from mlagents_envs.base_env import _ActionTupleBase


class LogProbsTuple(_ActionTupleBase):
    """
    An object whose fields correspond to the log probs of actions of different types.
    Continuous and discrete are numpy arrays
    Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size),
    respectively. Note, this also holds when continuous or discrete size is
    zero.
    """

    @property
    def discrete_dtype(self) -> np.dtype:
        """
        The dtype of a discrete log probability.
        """
        return np.float32


class ActionLogProbs(NamedTuple):
    """
    A NamedTuple containing the tensor for continuous log probs and list of tensors for
    discrete log probs of individual actions as well as all the log probs for an entire branch.
    Utility functions provide numpy <=> tensor conversions to be used by the optimizers.
    :param continuous_tensor: Torch tensor corresponding to log probs of continuous actions
    :param discrete_list: List of Torch tensors each corresponding to log probs of the discrete actions that were
    sampled.
    :param all_discrete_list: List of Torch tensors each corresponding to all log probs of
    a discrete action branch, even the discrete actions that were not sampled. all_discrete_list is a list of Tensors,
    each Tensor corresponds to one discrete branch log probabilities.
    """

    continuous_tensor: torch.Tensor
    discrete_list: Optional[List[torch.Tensor]]
    all_discrete_list: Optional[List[torch.Tensor]]

    @property
    def discrete_tensor(self):
        """
        Returns the discrete log probs list as a stacked tensor
        """
        return torch.stack(self.discrete_list, dim=-1)

    @property
    def all_discrete_tensor(self):
        """
        Returns the discrete log probs of each branch as a tensor
        """
        return torch.cat(self.all_discrete_list, dim=1)

    def to_log_probs_tuple(self) -> LogProbsTuple:
        """
        Returns a LogProbsTuple. Only adds if tensor is not None. Otherwise,
        LogProbsTuple uses a default.
        """
        log_probs_tuple = LogProbsTuple()
        if self.continuous_tensor is not None:
            continuous = ModelUtils.to_numpy(self.continuous_tensor)
            log_probs_tuple.add_continuous(continuous)
        if self.discrete_list is not None:
            discrete = ModelUtils.to_numpy(self.discrete_tensor)
            log_probs_tuple.add_discrete(discrete)
        return log_probs_tuple

    def _to_tensor_list(self) -> List[torch.Tensor]:
        """
        Returns the tensors in the ActionLogProbs as a flat List of torch Tensors. This
        is private and serves as a utility for self.flatten()
        """
        tensor_list: List[torch.Tensor] = []
        if self.continuous_tensor is not None:
            tensor_list.append(self.continuous_tensor)
        if self.discrete_list is not None:
            tensor_list.append(self.discrete_tensor)
        return tensor_list

    def flatten(self) -> torch.Tensor:
        """
        A utility method that returns all log probs in ActionLogProbs as a flattened tensor.
        This is useful for algorithms like PPO which can treat all log probs in the same way.
        """
        return torch.cat(self._to_tensor_list(), dim=1)

    @staticmethod
    def from_dict(buff: Dict[str, np.ndarray]) -> "ActionLogProbs":
        """
        A static method that accesses continuous and discrete log probs fields in an AgentBuffer
        and constructs the corresponding ActionLogProbs from the retrieved np arrays.
        """
        continuous: torch.Tensor = None
        discrete: List[torch.Tensor] = None  # type: ignore

        if "continuous_log_probs" in buff:
            continuous = ModelUtils.list_to_tensor(buff["continuous_log_probs"])
        if "discrete_log_probs" in buff:
            discrete_tensor = ModelUtils.list_to_tensor(buff["discrete_log_probs"])
            # This will keep discrete_list = None which enables flatten()
            if discrete_tensor.shape[1] > 0:
                discrete = [
                    discrete_tensor[..., i] for i in range(discrete_tensor.shape[-1])
                ]
        return ActionLogProbs(continuous, discrete, None)

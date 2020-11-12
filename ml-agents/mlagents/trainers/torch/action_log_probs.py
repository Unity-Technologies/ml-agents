from typing import List, Optional, NamedTuple, Dict
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.utils import ModelUtils


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

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns a Dict of np arrays with an entry correspinding to the continuous log probs
        and an entry corresponding to the discrete log probs. "continuous_log_probs" and
        "discrete_log_probs" are added to the agents buffer individually to maintain a flat buffer.
        """
        array_dict: Dict[str, np.ndarray] = {}
        if self.continuous_tensor is not None:
            array_dict["continuous_log_probs"] = ModelUtils.to_numpy(
                self.continuous_tensor
            )
        if self.discrete_list is not None:
            array_dict["discrete_log_probs"] = ModelUtils.to_numpy(self.discrete_tensor)
        return array_dict

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
            discrete = [
                discrete_tensor[..., i] for i in range(discrete_tensor.shape[-1])
            ]
        return ActionLogProbs(continuous, discrete, None)

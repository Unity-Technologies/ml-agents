from typing import Dict, Any
import torch

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy.nn_torch_policy import NNPolicy
from mlagents.trainers.optimizer import Optimizer


class TorchOptimizer(Optimizer):  # pylint: disable=W0223
    def __init__(self, policy: NNPolicy, trainer_params: Dict[str, Any]):
        super(TorchOptimizer, self).__init__()
        self.policy = policy
        self.trainer_params = trainer_params
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.memory_in: torch.Tensor = None
        self.memory_out: torch.Tensor = None
        self.m_size: int = 0
        self.global_step = torch.tensor(0)

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

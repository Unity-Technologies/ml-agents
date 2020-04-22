from typing import Dict, Any, Optional
import torch

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.components.bc.module import BCModule
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)


class TorchOptimizer(Optimizer):  # pylint: disable=W0223
    def __init__(self, policy: TorchPolicy, trainer_params: Dict[str, Any]):
        super(TorchOptimizer, self).__init__()
        self.policy = policy
        self.trainer_params = trainer_params
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.memory_in: torch.Tensor = None
        self.memory_out: torch.Tensor = None
        self.m_size: int = 0
        self.global_step = torch.tensor(0)
        self.bc_module: Optional[BCModule] = None

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        self.reward_signals = {}
        # Create reward signals
        for reward_signal, config in reward_signal_configs.items():
            self.reward_signals[reward_signal] = create_reward_signal(
                self.policy, reward_signal, config
            )
            self.update_dict.update(self.reward_signals[reward_signal].update_dict)

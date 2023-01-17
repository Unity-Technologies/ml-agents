from typing import Dict

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.settings import TrainerSettings


class TorchASEOptimizer(TorchPPOOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)

    def update_reward_signals(self, batch: AgentBuffer) -> Dict[str, float]:
        update_stats: Dict[str, float] = {}
        for name, reward_provider in self.reward_signals.items():
            if name == 'ase':
                update_stats.update(reward_provider.update(self.policy, batch))
            else:
                update_stats.update(reward_provider.update(batch))
        return update_stats

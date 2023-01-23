from typing import Dict

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.settings import TrainerSettings


class TorchASEOptimizer(TorchPPOOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        diversity_loss, diversity_stats = self.reward_signals['ase'].compute_diversity_loss(self.policy, batch)
        self.loss = self.reward_signals['ase'].diversity_objective_weight * diversity_loss
        update_stats = super().update(batch, num_sequences)
        return {**update_stats, **diversity_stats}

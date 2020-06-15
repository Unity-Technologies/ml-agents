import numpy as np

from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.buffer import AgentBuffer


class ExtrinsicRewardSignal(RewardSignal):
    def evaluate_batch(self, mini_batch: AgentBuffer) -> RewardSignalResult:
        env_rews = np.array(mini_batch["environment_rewards"], dtype=np.float32)
        return RewardSignalResult(self.strength * env_rews, env_rews)

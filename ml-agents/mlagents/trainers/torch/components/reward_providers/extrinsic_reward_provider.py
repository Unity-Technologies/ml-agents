import numpy as np
from typing import Dict

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)


class ExtrinsicRewardProvider(BaseRewardProvider):
    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        return np.array(mini_batch[BufferKey.ENVIRONMENT_REWARDS], dtype=np.float32)

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        return {}

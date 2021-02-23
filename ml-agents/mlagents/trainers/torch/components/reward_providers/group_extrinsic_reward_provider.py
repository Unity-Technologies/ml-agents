import numpy as np
from typing import Dict

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)


class GroupExtrinsicRewardProvider(BaseRewardProvider):
    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        indiv_rewards = np.array(
            mini_batch[BufferKey.ENVIRONMENT_REWARDS], dtype=np.float32
        )
        groupmate_rewards_list = mini_batch[BufferKey.GROUPMATE_REWARDS]
        groupmate_rewards_sum = np.array(
            [sum(_rew) for _rew in groupmate_rewards_list], dtype=np.ndarray
        )
        group_rewards = np.array(mini_batch[BufferKey.GROUP_REWARD], dtype=np.float32)
        # Add all the group rewards to the individual rewards
        return indiv_rewards + groupmate_rewards_sum + group_rewards

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        return {}

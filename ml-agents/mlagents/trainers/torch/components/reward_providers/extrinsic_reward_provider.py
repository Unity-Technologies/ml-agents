import numpy as np
from typing import Dict

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import ExtrinsicSettings


class ExtrinsicRewardProvider(BaseRewardProvider):
    """
    Evaluates extrinsic reward. For single-agent, this equals the individual reward
    given to the agent. For the COMA2 algorithm, we want not only the individual reward
    but also the team and the individual rewards of the other agents.
    """

    def __init__(self, specs: BehaviorSpec, settings: ExtrinsicSettings) -> None:
        super().__init__(specs, settings)
        self._add_groupmate_rewards = settings.add_groupmate_rewards

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        indiv_rewards = np.array(
            mini_batch[BufferKey.ENVIRONMENT_REWARDS], dtype=np.float32
        )
        total_rewards = indiv_rewards
        if (
            BufferKey.GROUPMATE_REWARDS in mini_batch
            and BufferKey.GROUP_REWARD in mini_batch
        ):
            if self._add_groupmate_rewards:
                groupmate_rewards_list = mini_batch[BufferKey.GROUPMATE_REWARDS]
                groupmate_rewards_sum = np.array(
                    [sum(_rew) for _rew in groupmate_rewards_list], dtype=np.float32
                )
                total_rewards += groupmate_rewards_sum
            group_rewards = np.array(
                mini_batch[BufferKey.GROUP_REWARD], dtype=np.float32
            )
            # Add all the group rewards to the individual rewards
            total_rewards += group_rewards
        return total_rewards

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        return {}

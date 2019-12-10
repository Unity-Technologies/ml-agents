from typing import Any, Dict, List
import numpy as np
from mlagents.trainers.brain import BrainInfo

from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult


class ExtrinsicRewardSignal(RewardSignal):
    @classmethod
    def check_config(
        cls, config_dict: Dict[str, Any], param_keys: List[str] = None
    ) -> None:
        """
        Checks the config and throw an exception if a hyperparameter is missing. Extrinsic requires strength and gamma
        at minimum.
        """
        param_keys = ["strength", "gamma"]
        super().check_config(config_dict, param_keys)

    def evaluate(
        self, current_info: BrainInfo, action: np.array, next_info: BrainInfo
    ) -> RewardSignalResult:
        """
        Evaluates the reward for the agents present in current_info given the next_info
        :param current_info: The current BrainInfo.
        :param next_info: The BrainInfo from the next timestep.
        :return: a RewardSignalResult of (scaled intrinsic reward, unscaled intrinsic reward) provided by the generator
        """
        unscaled_reward = np.array(next_info.rewards, dtype=np.float32)
        scaled_reward = self.strength * unscaled_reward
        return RewardSignalResult(scaled_reward, unscaled_reward)

    def evaluate_batch(self, mini_batch: Dict[str, np.array]) -> RewardSignalResult:
        env_rews = np.array(mini_batch["environment_rewards"], dtype=np.float32)
        return RewardSignalResult(self.strength * env_rews, env_rews)

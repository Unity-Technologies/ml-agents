from typing import Any, Dict, List
import numpy as np

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

    def evaluate_batch(self, mini_batch: Dict[str, np.array]) -> RewardSignalResult:
        env_rews = np.array(mini_batch["environment_rewards"], dtype=np.float32)
        return RewardSignalResult(self.strength * env_rews, env_rews)

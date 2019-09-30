from typing import Any, Dict, List
import numpy as np
from mlagents.envs.brain import BrainInfo

from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.models import LearningModel


class ExtrinsicRewardSignal(RewardSignal):
    def __init__(
        self,
        policy: TFPolicy,
        policy_model: LearningModel,
        strength: float,
        gamma: float,
    ):
        """
        The extrinsic reward generator. Returns the reward received by the environment
        :param policy: The Policy object (e.g. PPOPolicy) that this Reward Signal will apply to.
        :param strength: The strength of the reward. The reward's raw value will be multiplied by this value.
        :param gamma: The time discounting factor used for this reward.
        :return: An ExtrinsicRewardSignal object.
        """
        super().__init__(policy, policy_model, strength, gamma)

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
        self, current_info: BrainInfo, next_info: BrainInfo
    ) -> RewardSignalResult:
        """
        Evaluates the reward for the agents present in current_info given the next_info
        :param current_info: The current BrainInfo.
        :param next_info: The BrainInfo from the next timestep.
        :return: a RewardSignalResult of (scaled intrinsic reward, unscaled intrinsic reward) provided by the generator
        """
        unscaled_reward = np.array(next_info.rewards)
        scaled_reward = self.strength * unscaled_reward
        return RewardSignalResult(scaled_reward, unscaled_reward)

    def evaluate_batch(self, mini_batch: Dict[str, np.array]) -> RewardSignalResult:
        env_rews = np.array(mini_batch["environment_rewards"])
        return RewardSignalResult(self.strength * env_rews, env_rews)

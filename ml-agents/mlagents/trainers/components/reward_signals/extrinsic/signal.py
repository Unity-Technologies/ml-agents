import numpy as np

from mlagents.trainers.components.reward_signals import RewardSignal
from mlagents.trainers.policy import Policy


class ExtrinsicSignal(RewardSignal):
    def __init__(self, policy: Policy, strength, gamma):
        """
        The extrinsic reward generator. Returns the reward received by the environment
        :param signal_strength: The scaling parameter for the reward. The scaled reward will be the unscaled
        reward multiplied by the strength parameter
        """
        self.stat_name = "Environment/Extrinsic Reward"
        self.value_name = "Policy/Extrinsic Value Estimate"
        self.strength = strength
        self.gamma = gamma

    @classmethod
    def check_config(cls, config_dict):
        """
        Checks the config and throw an exception if a hyperparameter is missing. Extrinsic requires strength and gamma 
        at minimum. 
        """
        param_keys = ["strength", "gamma"]
        super().check_config(config_dict, param_keys)

    def evaluate(self, current_info, next_info):
        unscaled_reward = np.array(next_info.rewards)
        scaled_reward = self.strength * unscaled_reward
        return scaled_reward, unscaled_reward

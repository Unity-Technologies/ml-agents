import numpy as np

from mlagents.trainers.ppo.reward_signals import RewardSignal


class ExtrinsicSignal(RewardSignal):
    def __init__(self, signal_strength):
        self.stat_name = 'Environment/Extrinsic Reward'
        self.value_name = 'Policy/Extrinsic Value Estimate'
        self.strength = signal_strength

    def evaluate(self, current_info, next_info):
        unscaled_reward = np.array(next_info.rewards)
        scaled_reward = self.strength * unscaled_reward
        return scaled_reward, unscaled_reward

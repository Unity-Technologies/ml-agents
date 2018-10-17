import numpy as np

from mlagents.trainers.ppo.reward_signals import RewardSignal


class ExtrinsicSignal(RewardSignal):
    def __init__(self, signal_strength):
        self.stat_name = 'Environment/Cumulative Reward'
        self.strength = signal_strength

    def evaluate(self, current_info, next_info):
        return np.array(next_info.rewards) * self.strength

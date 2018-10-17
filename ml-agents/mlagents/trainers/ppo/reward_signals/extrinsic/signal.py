from mlagents.trainers.ppo.reward_signals import RewardSignal


class ExtrinsicSignal(RewardSignal):
    def __init__(self):
        self.stat_name = 'Environment/Cumulative Reward'

    def evaluate(self, current_info, next_info):
        return next_info.rewards

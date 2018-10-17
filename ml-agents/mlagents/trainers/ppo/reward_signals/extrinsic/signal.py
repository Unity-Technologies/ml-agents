from mlagents.trainers.ppo.reward_signals.reward_signal import RewardSignal


class ExtrinsicSignal(RewardSignal):
    def __init__(self, stat_name):
        self.stat_name = stat_name

    def evaluate(self, current_info, next_info):
        return next_info.rewards

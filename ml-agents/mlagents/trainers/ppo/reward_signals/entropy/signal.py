from mlagents.trainers.ppo.reward_signals import RewardSignal


class EntropySignal(RewardSignal):
    def __init__(self, policy, signal_strength):
        self.policy = policy
        self.strength = signal_strength
        self.stat_name = 'Policy/Entropy Reward'

    def evaluate(self, current_info, next_info):
        run_out = self.policy.evaluate(current_info)
        return self.strength * run_out['entropy']

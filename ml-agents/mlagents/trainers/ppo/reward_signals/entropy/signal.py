from mlagents.trainers.ppo.reward_signals import RewardSignal


class EntropySignal(RewardSignal):
    def __init__(self, policy, signal_strength):
        self.policy = policy
        self.strength = signal_strength
        self.stat_name = 'Policy/Entropy Reward'
        self.value_name = 'Policy/Entropy Value Estimate'

    def evaluate(self, current_info, next_info):
        run_out = self.policy.evaluate(current_info)
        unscaled_reward = run_out['entropy']
        scaled_reward = self.strength * unscaled_reward
        return scaled_reward, unscaled_reward

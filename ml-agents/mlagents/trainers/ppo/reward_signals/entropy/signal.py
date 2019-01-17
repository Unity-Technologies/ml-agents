from mlagents.trainers.ppo.reward_signals import RewardSignal
from mlagents.trainers.policy import Policy


class EntropySignal(RewardSignal):
    def __init__(self, policy: Policy, signal_strength):
        """
        The Entropy signal generator. The reward corresponds to the Entropy of the decision made
        :param policy: The policy of the learning parameter
        :param signal_strength: The scaling parameter for the reward. The scaled reward will be the unscaled
        reward multiplied by the strength parameter
        """
        self.policy = policy
        self.strength = signal_strength
        self.stat_name = 'Policy/Entropy Reward'
        self.value_name = 'Policy/Entropy Value Estimate'

    def evaluate(self, current_info, next_info):
        run_out = self.policy.evaluate(current_info)
        unscaled_reward = run_out['entropy']
        scaled_reward = self.strength * unscaled_reward
        return scaled_reward, unscaled_reward

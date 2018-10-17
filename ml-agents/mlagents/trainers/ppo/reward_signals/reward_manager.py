class RewardManager(object):
    def __init__(self, generator, stat_name):
        self.generator = generator
        self.agents_rewards = {}
        self.stat_name = stat_name

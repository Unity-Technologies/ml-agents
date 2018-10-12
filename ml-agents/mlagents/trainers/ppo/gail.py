import numpy as np

from .intrinsic_reward import IntrinsicReward
from .gail_discriminator import Discriminator
from ..demo_loader import demo_to_buffer


class GAIL(IntrinsicReward):
    def __init__(self, o_size, a_size, h_size, lr, demo_path):
        super().__init__()
        self.name = "GAIL"
        self.discriminator = Discriminator(o_size, a_size, h_size, lr)
        self.expert_demos, _ = demo_to_buffer(demo_path, 1)

    def get_reward(self, current_infos):
        return np.ones(len(current_infos))

    def update_generator(self, mini_batch):
        return None

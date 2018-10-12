import logging
from mlagents.trainers.trainer import UnityTrainerException

logger = logging.getLogger("mlagents.trainers")


class IntrinsicReward(object):
    def __init__(self):
        self.name = "Base"

    def get_reward(self, current_infos):
        """
        Returns the number of training steps the trainer has performed
        :return: the intrinsic reward provided by the generator
        """
        raise UnityTrainerException("The get_reward property was not implemented.")

    def update_generator(self, mini_batch):
        raise UnityTrainerException("The update_generator property was not implemented.")

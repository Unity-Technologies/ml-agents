import logging
from mlagents.trainers.trainer import UnityTrainerException
import tensorflow as tf

logger = logging.getLogger("mlagents.trainers")


class RewardSignal(object):
    def evaluate(self, current_info, next_info):
        """
        Evaluates the reward for the agents present in current_info given the next_info
        :return: a tuple of (scaled intrinsic reward, unscaled intrinsic reward) provided by the generator
        """
        raise UnityTrainerException("The evaluate function was not implemented.")

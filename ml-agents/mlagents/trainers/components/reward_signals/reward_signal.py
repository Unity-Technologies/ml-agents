import logging
from mlagents.trainers.trainer import UnityTrainerException
from mlagents.trainers.policy import Policy
import abc

import tensorflow as tf

logger = logging.getLogger("mlagents.trainers")


class RewardSignal(object):
    @abc.abstractmethod
    def __init__(self, policy: Policy, strength, gamma):
        """
        Initializes a reward signal. At minimum, you must pass in the policy it is being applied to, 
        the reward strength, and the gamma (discount factor.)
        """

    @abc.abstractmethod
    def evaluate(self, current_info, next_info):
        """
        Evaluates the reward for the agents present in current_info given the next_info
        :return: a tuple of (scaled intrinsic reward, unscaled intrinsic reward) provided by the generator
        """
        raise UnityTrainerException("The evaluate function was not implemented.")

    @classmethod
    def check_config(cls, config_dict, param_keys=[]):
        """
        Check the config dict, and throw an error if there are missing hyperparameters.
        """
        for k in param_keys:
            if k not in config_dict:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for {1}.".format(
                        k, cls.__name__
                    )
                )

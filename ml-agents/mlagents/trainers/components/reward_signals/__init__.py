import logging
from typing import Any, Dict, List
from collections import namedtuple
import numpy as np
import abc

from mlagents.tf_utils import tf

from mlagents.trainers.trainer import UnityTrainerException
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.models import LearningModel

logger = logging.getLogger("mlagents.trainers")

RewardSignalResult = namedtuple(
    "RewardSignalResult", ["scaled_reward", "unscaled_reward"]
)


class RewardSignal(abc.ABC):
    def __init__(
        self,
        policy: TFPolicy,
        policy_model: LearningModel,
        strength: float,
        gamma: float,
    ):
        """
        Initializes a reward signal. At minimum, you must pass in the policy it is being applied to,
        the reward strength, and the gamma (discount factor.)
        :param policy: The Policy object (e.g. PPOPolicy) that this Reward Signal will apply to.
        :param strength: The strength of the reward. The reward's raw value will be multiplied by this value.
        :param gamma: The time discounting factor used for this reward.
        :return: A RewardSignal object.
        """
        class_name = self.__class__.__name__
        short_name = class_name.replace("RewardSignal", "")
        self.stat_name = f"Policy/{short_name} Reward"
        self.value_name = f"Policy/{short_name} Value Estimate"
        # Terminate discounted reward computation at Done. Can disable to mitigate positive bias in rewards with
        # no natural end, e.g. GAIL or Curiosity
        self.use_terminal_states = True
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.gamma = gamma
        self.policy = policy
        self.policy_model = policy_model
        self.strength = strength
        self.stats_name_to_update_name: Dict[str, str] = {}

    def evaluate_batch(self, mini_batch: Dict[str, np.array]) -> RewardSignalResult:
        """
        Evaluates the reward for the data present in the Dict mini_batch. Use this when evaluating a reward
        function drawn straight from a Buffer.
        :param mini_batch: A Dict of numpy arrays (the format used by our Buffer)
            when drawing from the update buffer.
        :return: a RewardSignalResult of (scaled intrinsic reward, unscaled intrinsic reward) provided by the generator
        """
        mini_batch_len = len(next(iter(mini_batch.values())))
        return RewardSignalResult(
            self.strength * np.zeros(mini_batch_len, dtype=np.float32),
            np.zeros(mini_batch_len, dtype=np.float32),
        )

    def prepare_update(
        self,
        policy_model: LearningModel,
        mini_batch: Dict[str, np.ndarray],
        num_sequences: int,
    ) -> Dict[tf.Tensor, Any]:
        """
        If the reward signal has an internal model (e.g. GAIL or Curiosity), get the feed_dict
        needed to update the buffer..
        :param update_buffer: An AgentBuffer that contains the live data from which to update.
        :param n_sequences: The number of sequences in the training buffer.
        :return: A dict that corresponds to the feed_dict needed for the update.
        """
        return {}

    @classmethod
    def check_config(
        cls, config_dict: Dict[str, Any], param_keys: List[str] = None
    ) -> None:
        """
        Check the config dict, and throw an error if there are missing hyperparameters.
        """
        param_keys = param_keys or []
        for k in param_keys:
            if k not in config_dict:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for {1}.".format(
                        k, cls.__name__
                    )
                )

from typing import Any, Dict
from collections import namedtuple
import numpy as np
import abc

from mlagents.tf_utils import tf

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.settings import RewardSignalSettings


logger = get_logger(__name__)

RewardSignalResult = namedtuple(
    "RewardSignalResult", ["scaled_reward", "unscaled_reward"]
)


class RewardSignal(abc.ABC):
    def __init__(self, policy: TFPolicy, settings: RewardSignalSettings):
        """
        Initializes a reward signal. At minimum, you must pass in the policy it is being applied to,
        the reward strength, and the gamma (discount factor.)
        :param policy: The Policy object (e.g. NNPolicy) that this Reward Signal will apply to.
        :param settings: Settings parameters for this Reward Signal, including gamma and strength.
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
        self.gamma = settings.gamma
        self.policy = policy
        self.strength = settings.strength
        self.stats_name_to_update_name: Dict[str, str] = {}

    def evaluate_batch(self, mini_batch: AgentBuffer) -> RewardSignalResult:
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
        self, policy: TFPolicy, mini_batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        """
        If the reward signal has an internal model (e.g. GAIL or Curiosity), get the feed_dict
        needed to update the buffer..
        :param update_buffer: An AgentBuffer that contains the live data from which to update.
        :param n_sequences: The number of sequences in the training buffer.
        :return: A dict that corresponds to the feed_dict needed for the update.
        """
        return {}

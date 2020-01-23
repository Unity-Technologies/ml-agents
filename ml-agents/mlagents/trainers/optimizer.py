import abc
from typing import Dict, Any
from tf_utils import tf

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy import Policy
from mlagents.trainers.tf_policy import TFPolicy


class Optimizer(abc.ABC):
    """
    Creates loss functions and auxillary networks (e.g. Q or Value) needed for training.
    Provides methods to update the Policy.
    """

    def __init__(self, policy: Policy, optimizer_parameters: Dict[str, Any]):
        """
        Create loss functions and auxillary networks.
        """

    @abc.abstractmethod
    def update_batch(self, batch: AgentBuffer):
        """
        Update the Policy based on the batch that was passed in.
        """


class TFOptimizer(Optimizer):
    def __init__(self, sess: tf.Session, policy: TFPolicy, reward_signal_configs):
        self.sess = sess
        self.policy = policy
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.value_heads: Dict[str, tf.Tensor] = {}
        self.create_reward_signals(reward_signal_configs)

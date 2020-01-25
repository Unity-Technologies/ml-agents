import abc
from typing import Dict, Any, List
import numpy as np

from mlagents.tf_utils.tf import tf
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.policy import Policy
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)


class Optimizer(abc.ABC):
    """
    Creates loss functions and auxillary networks (e.g. Q or Value) needed for training.
    Provides methods to update the Policy.
    """

    @abc.abstractmethod
    def __init__(self, policy: Policy):
        """
        Create loss functions and auxillary networks.
        """
        pass

    @abc.abstractmethod
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Update the Policy based on the batch that was passed in.
        """
        pass


class TFOptimizer(Optimizer, abc.ABC):  # pylint: disable=W0223
    def __init__(self, policy: TFPolicy, trainer_params: Dict[str, Any]):
        super().__init__(policy)
        self.sess = policy.sess
        self.policy = policy
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.value_heads: Dict[str, tf.Tensor] = {}
        self.create_reward_signals(trainer_params["reward_signals"])

    def get_batched_value_estimates(self, batch: AgentBuffer) -> Dict[str, np.ndarray]:
        feed_dict: Dict[tf.Tensor, Any] = {
            self.policy.batch_size_ph: batch.num_experiences,
            self.policy.sequence_length_ph: 1,  # We want to feed data in batch-wise, not time-wise.
        }

        if self.policy.vec_obs_size > 0:
            feed_dict[self.policy.vector_in] = batch["vector_obs"]
        if self.policy.vis_obs_size > 0:
            for i in range(len(self.policy.visual_in)):
                _obs = batch["visual_obs%d" % i]
                feed_dict[self.policy.visual_in[i]] = _obs
        if self.policy.use_recurrent:
            feed_dict[self.policy.memory_in] = batch["memory"]
        if self.policy.prev_action is not None:
            feed_dict[self.policy.prev_action] = batch["prev_action"]
        value_estimates = self.sess.run(self.value_heads, feed_dict)
        value_estimates = {k: np.squeeze(v, axis=1) for k, v in value_estimates.items()}

        return value_estimates

    def get_value_estimates(
        self, next_obs: List[np.ndarray], agent_id: str, done: bool
    ) -> Dict[str, float]:
        """
        Generates value estimates for bootstrapping.
        :param experience: AgentExperience to be used for bootstrapping.
        :param done: Whether or not this is the last element of the episode, in which case the value estimate will be 0.
        :return: The value estimate dictionary with key being the name of the reward signal and the value the
        corresponding value estimate.
        """

        feed_dict: Dict[tf.Tensor, Any] = {
            self.policy.batch_size_ph: 1,
            self.policy.sequence_length_ph: 1,
        }
        vec_vis_obs = SplitObservations.from_observations(next_obs)
        for i in range(len(vec_vis_obs.visual_observations)):
            feed_dict[self.policy.visual_in[i]] = [vec_vis_obs.visual_observations[i]]

        if self.policy.vec_obs_size > 0:
            feed_dict[self.policy.vector_in] = [vec_vis_obs.vector_observations]
        # if self.policy.use_recurrent:
        #     feed_dict[self.policy.memory_in] = self.policy.retrieve_memories([agent_id])
        # if self.policy.prev_action is not None:
        #     feed_dict[self.policy.prev_action] = self.policy.retrieve_previous_action(
        #         [agent_id]
        #     )
        value_estimates = self.sess.run(self.value_heads, feed_dict)

        value_estimates = {k: float(v) for k, v in value_estimates.items()}

        # If we're done, reassign all of the value estimates that need terminal states.
        if done:
            for k in value_estimates:
                if self.reward_signals[k].use_terminal_states:
                    value_estimates[k] = 0.0

        return value_estimates

    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        self.reward_signals = {}
        # Create reward signals
        for reward_signal, config in reward_signal_configs.items():
            self.reward_signals[reward_signal] = create_reward_signal(
                self, self.policy, reward_signal, config
            )
            self.update_dict.update(self.reward_signals[reward_signal].update_dict)

    def create_value_heads(self, stream_names, hidden_input):
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        for name in stream_names:
            value = tf.layers.dense(hidden_input, 1, name="{}_value".format(name))
            self.value_heads[name] = value
        self.value = tf.reduce_mean(list(self.value_heads.values()), 0)

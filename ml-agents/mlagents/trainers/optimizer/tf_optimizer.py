from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from mlagents.tf_utils.tf import tf
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)
from mlagents.trainers.components.bc.module import BCModule


class TFOptimizer(Optimizer):  # pylint: disable=W0223
    def __init__(self, policy: TFPolicy, trainer_params: Dict[str, Any]):
        self.sess = policy.sess
        self.policy = policy
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.value_heads: Dict[str, tf.Tensor] = {}
        self.create_reward_signals(trainer_params["reward_signals"])
        self.memory_in: tf.Tensor = None
        self.memory_out: tf.Tensor = None
        self.m_size: int = 0
        self.bc_module: Optional[BCModule] = None
        # Create pretrainer if needed
        if "behavioral_cloning" in trainer_params:
            BCModule.check_config(trainer_params["behavioral_cloning"])
            self.bc_module = BCModule(
                self.policy,
                policy_learning_rate=trainer_params["learning_rate"],
                default_batch_size=trainer_params["batch_size"],
                default_num_epoch=3,
                **trainer_params["behavioral_cloning"],
            )

    def get_trajectory_value_estimates(
        self, batch: AgentBuffer, next_obs: List[np.ndarray], done: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        feed_dict: Dict[tf.Tensor, Any] = {
            self.policy.batch_size_ph: batch.num_experiences,
            self.policy.sequence_length_ph: batch.num_experiences,  # We want to feed data in batch-wise, not time-wise.
        }

        if self.policy.vec_obs_size > 0:
            feed_dict[self.policy.vector_in] = batch["vector_obs"]
        if self.policy.vis_obs_size > 0:
            for i in range(len(self.policy.visual_in)):
                _obs = batch["visual_obs%d" % i]
                feed_dict[self.policy.visual_in[i]] = _obs
        if self.policy.use_recurrent:
            feed_dict[self.policy.memory_in] = [
                np.zeros((self.policy.m_size), dtype=np.float32)
            ]
            feed_dict[self.memory_in] = [np.zeros((self.m_size), dtype=np.float32)]
        if self.policy.prev_action is not None:
            feed_dict[self.policy.prev_action] = batch["prev_action"]

        if self.policy.use_recurrent:
            value_estimates, policy_mem, value_mem = self.sess.run(
                [self.value_heads, self.policy.memory_out, self.memory_out], feed_dict
            )
            prev_action = (
                batch["actions"][-1] if not self.policy.use_continuous_act else None
            )
        else:
            value_estimates = self.sess.run(self.value_heads, feed_dict)
            prev_action = None
            policy_mem = None
            value_mem = None
        value_estimates = {k: np.squeeze(v, axis=1) for k, v in value_estimates.items()}

        # We do this in a separate step to feed the memory outs - a further optimization would
        # be to append to the obs before running sess.run.
        final_value_estimates = self._get_value_estimates(
            next_obs, done, policy_mem, value_mem, prev_action
        )

        return value_estimates, final_value_estimates

    def _get_value_estimates(
        self,
        next_obs: List[np.ndarray],
        done: bool,
        policy_memory: np.ndarray = None,
        value_memory: np.ndarray = None,
        prev_action: np.ndarray = None,
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
        if policy_memory is not None:
            feed_dict[self.policy.memory_in] = policy_memory
        if value_memory is not None:
            feed_dict[self.memory_in] = value_memory
        if prev_action is not None:
            feed_dict[self.policy.prev_action] = [prev_action]
        value_estimates = self.sess.run(self.value_heads, feed_dict)

        value_estimates = {k: float(v) for k, v in value_estimates.items()}

        # If we're done, reassign all of the value estimates that need terminal states.
        if done:
            for k in value_estimates:
                if self.reward_signals[k].use_terminal_states:
                    value_estimates[k] = 0.0

        return value_estimates

    def create_reward_signals(self, reward_signal_configs: Dict[str, Any]) -> None:
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        self.reward_signals = {}
        # Create reward signals
        for reward_signal, config in reward_signal_configs.items():
            self.reward_signals[reward_signal] = create_reward_signal(
                self.policy, reward_signal, config
            )
            self.update_dict.update(self.reward_signals[reward_signal].update_dict)

    def create_optimizer_op(
        self, learning_rate: tf.Tensor, name: str = "Adam"
    ) -> tf.train.Optimizer:
        return tf.train.AdamOptimizer(learning_rate=learning_rate, name=name)

    def _execute_model(
        self, feed_dict: Dict[tf.Tensor, np.ndarray], out_dict: Dict[str, tf.Tensor]
    ) -> Dict[str, np.ndarray]:
        """
        Executes model.
        :param feed_dict: Input dictionary mapping nodes to input data.
        :param out_dict: Output dictionary mapping names to nodes.
        :return: Dictionary mapping names to input data.
        """
        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

    def _make_zero_mem(self, m_size: int, length: int) -> List[np.ndarray]:
        return [
            np.zeros((m_size), dtype=np.float32)
            for i in range(0, length, self.policy.sequence_length)
        ]

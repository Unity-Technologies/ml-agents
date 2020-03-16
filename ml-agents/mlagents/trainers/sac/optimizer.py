import logging
import numpy as np
from typing import Dict, List, Optional, Any, Mapping

from mlagents.tf_utils import tf

from mlagents.trainers.sac.network import SACPolicyNetwork, SACTargetNetwork
from mlagents.trainers.models import LearningRateSchedule, EncoderType, ModelUtils
from mlagents.trainers.optimizer.tf_optimizer import TFOptimizer
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.timers import timed

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = logging.getLogger("mlagents.trainers")

POLICY_SCOPE = ""
TARGET_SCOPE = "target_network"


class SACOptimizer(TFOptimizer):
    def __init__(self, policy: TFPolicy, trainer_params: Dict[str, Any]):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: Brain parameters used to generate specific network graph.
        :param lr: Learning rate.
        :param lr_schedule: Learning rate decay schedule.
        :param h_size: Size of hidden layers
        :param init_entcoef: Initial value for entropy coefficient. Set lower to learn faster,
            set higher to explore more.
        :return: a sub-class of PPOAgent tailored to the environment.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers: Number of hidden layers between encoded input and policy & value layers
        :param tau: Strength of soft-Q update.
        :param m_size: Size of brain memory.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.
        policy.create_tf_graph()

        with policy.graph.as_default():
            with tf.variable_scope(""):
                super().__init__(policy, trainer_params)
                lr = float(trainer_params["learning_rate"])
                lr_schedule = LearningRateSchedule(
                    trainer_params.get("learning_rate_schedule", "constant")
                )
                self.policy = policy
                self.act_size = self.policy.act_size
                h_size = int(trainer_params["hidden_units"])
                max_step = float(trainer_params["max_steps"])
                num_layers = int(trainer_params["num_layers"])
                vis_encode_type = EncoderType(
                    trainer_params.get("vis_encode_type", "simple")
                )
                self.tau = trainer_params.get("tau", 0.005)
                self.burn_in_ratio = float(trainer_params.get("burn_in_ratio", 0.0))

                # Non-exposed SAC parameters
                self.discrete_target_entropy_scale = (
                    0.2
                )  # Roughly equal to e-greedy 0.05
                self.continuous_target_entropy_scale = 1.0

                self.init_entcoef = trainer_params.get("init_entcoef", 1.0)
                stream_names = list(self.reward_signals.keys())
                # Use to reduce "survivor bonus" when using Curiosity or GAIL.
                self.gammas = [
                    _val["gamma"] for _val in trainer_params["reward_signals"].values()
                ]
                self.use_dones_in_backup = {
                    name: tf.Variable(1.0) for name in stream_names
                }
                self.disable_use_dones = {
                    name: self.use_dones_in_backup[name].assign(0.0)
                    for name in stream_names
                }

                if num_layers < 1:
                    num_layers = 1

                self.target_init_op: List[tf.Tensor] = []
                self.target_update_op: List[tf.Tensor] = []
                self.update_batch_policy: Optional[tf.Operation] = None
                self.update_batch_value: Optional[tf.Operation] = None
                self.update_batch_entropy: Optional[tf.Operation] = None

                self.policy_network = SACPolicyNetwork(
                    policy=self.policy,
                    m_size=self.policy.m_size,  # 3x policy.m_size
                    h_size=h_size,
                    normalize=self.policy.normalize,
                    use_recurrent=self.policy.use_recurrent,
                    num_layers=num_layers,
                    stream_names=stream_names,
                    vis_encode_type=vis_encode_type,
                )
                self.target_network = SACTargetNetwork(
                    policy=self.policy,
                    m_size=self.policy.m_size,  # 1x policy.m_size
                    h_size=h_size,
                    normalize=self.policy.normalize,
                    use_recurrent=self.policy.use_recurrent,
                    num_layers=num_layers,
                    stream_names=stream_names,
                    vis_encode_type=vis_encode_type,
                )
                # The optimizer's m_size is 3 times the policy (Q1, Q2, and Value)
                self.m_size = 3 * self.policy.m_size
                self._create_inputs_and_outputs()
                self.learning_rate = ModelUtils.create_learning_rate(
                    lr_schedule, lr, self.policy.global_step, int(max_step)
                )
                self._create_losses(
                    self.policy_network.q1_heads,
                    self.policy_network.q2_heads,
                    lr,
                    int(max_step),
                    stream_names,
                    discrete=not self.policy.use_continuous_act,
                )
                self._create_sac_optimizer_ops()

                self.selected_actions = (
                    self.policy.selected_actions
                )  # For GAIL and other reward signals
                if self.policy.normalize:
                    target_update_norm = self.target_network.copy_normalization(
                        self.policy.running_mean,
                        self.policy.running_variance,
                        self.policy.normalization_steps,
                    )
                    # Update the normalization of the optimizer when the policy does.
                    self.policy.update_normalization_op = tf.group(
                        [self.policy.update_normalization_op, target_update_norm]
                    )

                self.policy.initialize_or_load()

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
            "Losses/Q1 Loss": "q1_loss",
            "Losses/Q2 Loss": "q2_loss",
            "Policy/Entropy Coeff": "entropy_coef",
            "Policy/Learning Rate": "learning_rate",
        }

        self.update_dict = {
            "value_loss": self.total_value_loss,
            "policy_loss": self.policy_loss,
            "q1_loss": self.q1_loss,
            "q2_loss": self.q2_loss,
            "entropy_coef": self.ent_coef,
            "entropy": self.policy.entropy,
            "update_batch": self.update_batch_policy,
            "update_value": self.update_batch_value,
            "update_entropy": self.update_batch_entropy,
            "learning_rate": self.learning_rate,
        }

    def _create_inputs_and_outputs(self) -> None:
        """
        Assign the higher-level SACModel's inputs and outputs to those of its policy or
        target network.
        """
        self.vector_in = self.policy.vector_in
        self.visual_in = self.policy.visual_in
        self.next_vector_in = self.target_network.vector_in
        self.next_visual_in = self.target_network.visual_in
        self.sequence_length_ph = self.policy.sequence_length_ph
        self.next_sequence_length_ph = self.target_network.sequence_length_ph
        if not self.policy.use_continuous_act:
            self.action_masks = self.policy_network.action_masks
        else:
            self.output_pre = self.policy_network.output_pre

        # Don't use value estimate during inference.
        self.value = tf.identity(
            self.policy_network.value, name="value_estimate_unused"
        )
        self.value_heads = self.policy_network.value_heads
        self.dones_holder = tf.placeholder(
            shape=[None], dtype=tf.float32, name="dones_holder"
        )

        if self.policy.use_recurrent:
            self.memory_in = self.policy_network.memory_in
            self.memory_out = self.policy_network.memory_out
            if not self.policy.use_continuous_act:
                self.prev_action = self.policy_network.prev_action
            self.next_memory_in = self.target_network.memory_in

    def _create_losses(
        self,
        q1_streams: Dict[str, tf.Tensor],
        q2_streams: Dict[str, tf.Tensor],
        lr: tf.Tensor,
        max_step: int,
        stream_names: List[str],
        discrete: bool = False,
    ) -> None:
        """
        Creates training-specific Tensorflow ops for SAC models.
        :param q1_streams: Q1 streams from policy network
        :param q1_streams: Q2 streams from policy network
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        :param stream_names: List of reward stream names.
        :param discrete: Whether or not to use discrete action losses.
        """

        if discrete:
            self.target_entropy = [
                self.discrete_target_entropy_scale * np.log(i).astype(np.float32)
                for i in self.act_size
            ]
            discrete_action_probs = tf.exp(self.policy.all_log_probs)
            per_action_entropy = discrete_action_probs * self.policy.all_log_probs
        else:
            self.target_entropy = (
                -1
                * self.continuous_target_entropy_scale
                * np.prod(self.act_size[0]).astype(np.float32)
            )

        self.rewards_holders = {}
        self.min_policy_qs = {}

        for name in stream_names:
            if discrete:
                _branched_mpq1 = ModelUtils.break_into_branches(
                    self.policy_network.q1_pheads[name] * discrete_action_probs,
                    self.act_size,
                )
                branched_mpq1 = tf.stack(
                    [
                        tf.reduce_sum(_br, axis=1, keep_dims=True)
                        for _br in _branched_mpq1
                    ]
                )
                _q1_p_mean = tf.reduce_mean(branched_mpq1, axis=0)

                _branched_mpq2 = ModelUtils.break_into_branches(
                    self.policy_network.q2_pheads[name] * discrete_action_probs,
                    self.act_size,
                )
                branched_mpq2 = tf.stack(
                    [
                        tf.reduce_sum(_br, axis=1, keep_dims=True)
                        for _br in _branched_mpq2
                    ]
                )
                _q2_p_mean = tf.reduce_mean(branched_mpq2, axis=0)

                self.min_policy_qs[name] = tf.minimum(_q1_p_mean, _q2_p_mean)
            else:
                self.min_policy_qs[name] = tf.minimum(
                    self.policy_network.q1_pheads[name],
                    self.policy_network.q2_pheads[name],
                )

            rewards_holder = tf.placeholder(
                shape=[None], dtype=tf.float32, name="{}_rewards".format(name)
            )
            self.rewards_holders[name] = rewards_holder

        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        expanded_dones = tf.expand_dims(self.dones_holder, axis=-1)
        for i, name in enumerate(stream_names):
            _expanded_rewards = tf.expand_dims(self.rewards_holders[name], axis=-1)

            q_backup = tf.stop_gradient(
                _expanded_rewards
                + (1.0 - self.use_dones_in_backup[name] * expanded_dones)
                * self.gammas[i]
                * self.target_network.value_heads[name]
            )

            if discrete:
                # We need to break up the Q functions by branch, and update them individually.
                branched_q1_stream = ModelUtils.break_into_branches(
                    self.policy.selected_actions * q1_streams[name], self.act_size
                )
                branched_q2_stream = ModelUtils.break_into_branches(
                    self.policy.selected_actions * q2_streams[name], self.act_size
                )

                # Reduce each branch into scalar
                branched_q1_stream = [
                    tf.reduce_sum(_branch, axis=1, keep_dims=True)
                    for _branch in branched_q1_stream
                ]
                branched_q2_stream = [
                    tf.reduce_sum(_branch, axis=1, keep_dims=True)
                    for _branch in branched_q2_stream
                ]

                q1_stream = tf.reduce_mean(branched_q1_stream, axis=0)
                q2_stream = tf.reduce_mean(branched_q2_stream, axis=0)

            else:
                q1_stream = q1_streams[name]
                q2_stream = q2_streams[name]

            _q1_loss = 0.5 * tf.reduce_mean(
                tf.to_float(self.policy.mask)
                * tf.squared_difference(q_backup, q1_stream)
            )

            _q2_loss = 0.5 * tf.reduce_mean(
                tf.to_float(self.policy.mask)
                * tf.squared_difference(q_backup, q2_stream)
            )

            q1_losses.append(_q1_loss)
            q2_losses.append(_q2_loss)

        self.q1_loss = tf.reduce_mean(q1_losses)
        self.q2_loss = tf.reduce_mean(q2_losses)

        # Learn entropy coefficient
        if discrete:
            # Create a log_ent_coef for each branch
            self.log_ent_coef = tf.get_variable(
                "log_ent_coef",
                dtype=tf.float32,
                initializer=np.log([self.init_entcoef] * len(self.act_size)).astype(
                    np.float32
                ),
                trainable=True,
            )
        else:
            self.log_ent_coef = tf.get_variable(
                "log_ent_coef",
                dtype=tf.float32,
                initializer=np.log(self.init_entcoef).astype(np.float32),
                trainable=True,
            )

        self.ent_coef = tf.exp(self.log_ent_coef)
        if discrete:
            # We also have to do a different entropy and target_entropy per branch.
            branched_per_action_ent = ModelUtils.break_into_branches(
                per_action_entropy, self.act_size
            )
            branched_ent_sums = tf.stack(
                [
                    tf.reduce_sum(_lp, axis=1, keep_dims=True) + _te
                    for _lp, _te in zip(branched_per_action_ent, self.target_entropy)
                ],
                axis=1,
            )
            self.entropy_loss = -tf.reduce_mean(
                tf.to_float(self.policy.mask)
                * tf.reduce_mean(
                    self.log_ent_coef
                    * tf.squeeze(tf.stop_gradient(branched_ent_sums), axis=2),
                    axis=1,
                )
            )

            # Same with policy loss, we have to do the loss per branch and average them,
            # so that larger branches don't get more weight.
            # The equivalent KL divergence from Eq 10 of Haarnoja et al. is also pi*log(pi) - Q
            branched_q_term = ModelUtils.break_into_branches(
                discrete_action_probs * self.policy_network.q1_p, self.act_size
            )

            branched_policy_loss = tf.stack(
                [
                    tf.reduce_sum(self.ent_coef[i] * _lp - _qt, axis=1, keep_dims=True)
                    for i, (_lp, _qt) in enumerate(
                        zip(branched_per_action_ent, branched_q_term)
                    )
                ]
            )
            self.policy_loss = tf.reduce_mean(
                tf.to_float(self.policy.mask) * tf.squeeze(branched_policy_loss)
            )

            # Do vbackup entropy bonus per branch as well.
            branched_ent_bonus = tf.stack(
                [
                    tf.reduce_sum(self.ent_coef[i] * _lp, axis=1, keep_dims=True)
                    for i, _lp in enumerate(branched_per_action_ent)
                ]
            )
            value_losses = []
            for name in stream_names:
                v_backup = tf.stop_gradient(
                    self.min_policy_qs[name]
                    - tf.reduce_mean(branched_ent_bonus, axis=0)
                )
                value_losses.append(
                    0.5
                    * tf.reduce_mean(
                        tf.to_float(self.policy.mask)
                        * tf.squared_difference(
                            self.policy_network.value_heads[name], v_backup
                        )
                    )
                )

        else:
            self.entropy_loss = -tf.reduce_mean(
                self.log_ent_coef
                * tf.to_float(self.policy.mask)
                * tf.stop_gradient(
                    tf.reduce_sum(
                        self.policy.all_log_probs + self.target_entropy,
                        axis=1,
                        keep_dims=True,
                    )
                )
            )
            batch_policy_loss = tf.reduce_mean(
                self.ent_coef * self.policy.all_log_probs - self.policy_network.q1_p,
                axis=1,
            )
            self.policy_loss = tf.reduce_mean(
                tf.to_float(self.policy.mask) * batch_policy_loss
            )

            value_losses = []
            for name in stream_names:
                v_backup = tf.stop_gradient(
                    self.min_policy_qs[name]
                    - tf.reduce_sum(self.ent_coef * self.policy.all_log_probs, axis=1)
                )
                value_losses.append(
                    0.5
                    * tf.reduce_mean(
                        tf.to_float(self.policy.mask)
                        * tf.squared_difference(
                            self.policy_network.value_heads[name], v_backup
                        )
                    )
                )
        self.value_loss = tf.reduce_mean(value_losses)

        self.total_value_loss = self.q1_loss + self.q2_loss + self.value_loss

        self.entropy = self.policy_network.entropy

    def _create_sac_optimizer_ops(self) -> None:
        """
        Creates the Adam optimizers and update ops for SAC, including
        the policy, value, and entropy updates, as well as the target network update.
        """
        policy_optimizer = self.create_optimizer_op(
            learning_rate=self.learning_rate, name="sac_policy_opt"
        )
        entropy_optimizer = self.create_optimizer_op(
            learning_rate=self.learning_rate, name="sac_entropy_opt"
        )
        value_optimizer = self.create_optimizer_op(
            learning_rate=self.learning_rate, name="sac_value_opt"
        )

        self.target_update_op = [
            tf.assign(target, (1 - self.tau) * target + self.tau * source)
            for target, source in zip(
                self.target_network.value_vars, self.policy_network.value_vars
            )
        ]
        logger.debug("value_vars")
        self.print_all_vars(self.policy_network.value_vars)
        logger.debug("targvalue_vars")
        self.print_all_vars(self.target_network.value_vars)
        logger.debug("critic_vars")
        self.print_all_vars(self.policy_network.critic_vars)
        logger.debug("q_vars")
        self.print_all_vars(self.policy_network.q_vars)
        logger.debug("policy_vars")
        policy_vars = self.policy.get_trainable_variables()
        self.print_all_vars(policy_vars)

        self.target_init_op = [
            tf.assign(target, source)
            for target, source in zip(
                self.target_network.value_vars, self.policy_network.value_vars
            )
        ]

        self.update_batch_policy = policy_optimizer.minimize(
            self.policy_loss, var_list=policy_vars
        )

        # Make sure policy is updated first, then value, then entropy.
        with tf.control_dependencies([self.update_batch_policy]):
            self.update_batch_value = value_optimizer.minimize(
                self.total_value_loss, var_list=self.policy_network.critic_vars
            )
            # Add entropy coefficient optimization operation
            with tf.control_dependencies([self.update_batch_value]):
                self.update_batch_entropy = entropy_optimizer.minimize(
                    self.entropy_loss, var_list=self.log_ent_coef
                )

    def print_all_vars(self, variables):
        for _var in variables:
            logger.debug(_var)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param batch: Experience mini-batch.
        :param update_target: Whether or not to update target value network
        :param reward_signal_batches: Minibatches to use for updating the reward signals,
            indexed by name. If none, don't update the reward signals.
        :return: Output from update process.
        """
        feed_dict = self._construct_feed_dict(self.policy, batch, num_sequences)
        stats_needed = self.stats_name_to_update_name
        update_stats: Dict[str, float] = {}
        update_vals = self._execute_model(feed_dict, self.update_dict)
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        # Update target network. By default, target update happens at every policy update.
        self.sess.run(self.target_update_op)
        return update_stats

    def update_reward_signals(
        self, reward_signal_minibatches: Mapping[str, AgentBuffer], num_sequences: int
    ) -> Dict[str, float]:
        """
        Only update the reward signals.
        :param reward_signal_batches: Minibatches to use for updating the reward signals,
            indexed by name. If none, don't update the reward signals.
        """
        # Collect feed dicts for all reward signals.
        feed_dict: Dict[tf.Tensor, Any] = {}
        update_dict: Dict[str, tf.Tensor] = {}
        update_stats: Dict[str, float] = {}
        stats_needed: Dict[str, str] = {}
        if reward_signal_minibatches:
            self.add_reward_signal_dicts(
                feed_dict,
                update_dict,
                stats_needed,
                reward_signal_minibatches,
                num_sequences,
            )
        update_vals = self._execute_model(feed_dict, update_dict)
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        return update_stats

    def add_reward_signal_dicts(
        self,
        feed_dict: Dict[tf.Tensor, Any],
        update_dict: Dict[str, tf.Tensor],
        stats_needed: Dict[str, str],
        reward_signal_minibatches: Mapping[str, AgentBuffer],
        num_sequences: int,
    ) -> None:
        """
        Adds the items needed for reward signal updates to the feed_dict and stats_needed dict.
        :param feed_dict: Feed dict needed update
        :param update_dit: Update dict that needs update
        :param stats_needed: Stats needed to get from the update.
        :param reward_signal_minibatches: Minibatches to use for updating the reward signals,
            indexed by name.
        """
        for name, r_batch in reward_signal_minibatches.items():
            feed_dict.update(
                self.reward_signals[name].prepare_update(
                    self.policy, r_batch, num_sequences
                )
            )
            update_dict.update(self.reward_signals[name].update_dict)
            stats_needed.update(self.reward_signals[name].stats_name_to_update_name)

    def _construct_feed_dict(
        self, policy: TFPolicy, batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        """
        Builds the feed dict for updating the SAC model.
        :param model: The model to update. May be different when, e.g. using multi-GPU.
        :param batch: Mini-batch to use to update.
        :param num_sequences: Number of LSTM sequences in batch.
        """
        # Do an optional burn-in for memories
        num_burn_in = int(self.burn_in_ratio * self.policy.sequence_length)
        burn_in_mask = np.ones((self.policy.sequence_length), dtype=np.float32)
        burn_in_mask[range(0, num_burn_in)] = 0
        burn_in_mask = np.tile(burn_in_mask, num_sequences)
        feed_dict = {
            policy.batch_size_ph: num_sequences,
            policy.sequence_length_ph: self.policy.sequence_length,
            self.next_sequence_length_ph: self.policy.sequence_length,
            self.policy.mask_input: batch["masks"] * burn_in_mask,
        }
        for name in self.reward_signals:
            feed_dict[self.rewards_holders[name]] = batch["{}_rewards".format(name)]

        if self.policy.use_continuous_act:
            feed_dict[self.policy_network.external_action_in] = batch["actions"]
        else:
            feed_dict[policy.output] = batch["actions"]
            if self.policy.use_recurrent:
                feed_dict[policy.prev_action] = batch["prev_action"]
            feed_dict[policy.action_masks] = batch["action_mask"]
        if self.policy.use_vec_obs:
            feed_dict[policy.vector_in] = batch["vector_obs"]
            feed_dict[self.next_vector_in] = batch["next_vector_in"]
        if self.policy.vis_obs_size > 0:
            for i, _ in enumerate(policy.visual_in):
                _obs = batch["visual_obs%d" % i]
                feed_dict[policy.visual_in[i]] = _obs
            for i, _ in enumerate(self.next_visual_in):
                _obs = batch["next_visual_obs%d" % i]
                feed_dict[self.next_visual_in[i]] = _obs
        if self.policy.use_recurrent:
            feed_dict[policy.memory_in] = [
                batch["memory"][i]
                for i in range(0, len(batch["memory"]), self.policy.sequence_length)
            ]
            feed_dict[self.policy_network.memory_in] = self._make_zero_mem(
                self.m_size, batch.num_experiences
            )
            feed_dict[self.target_network.memory_in] = self._make_zero_mem(
                self.m_size // 3, batch.num_experiences
            )
        feed_dict[self.dones_holder] = batch["done"]
        return feed_dict

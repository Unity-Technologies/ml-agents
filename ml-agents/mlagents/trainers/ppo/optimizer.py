from typing import Optional, Any, Dict
import numpy as np
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed
from mlagents.trainers.models import ModelUtils, EncoderType, LearningRateSchedule
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.optimizer.tf_optimizer import TFOptimizer
from mlagents.trainers.buffer import AgentBuffer


class PPOOptimizer(TFOptimizer):
    def __init__(self, policy: TFPolicy, trainer_params: Dict[str, Any]):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TFPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.
        policy.create_tf_graph()

        with policy.graph.as_default():
            with tf.variable_scope("optimizer/"):
                super().__init__(policy, trainer_params)

                lr = float(trainer_params["learning_rate"])
                lr_schedule = LearningRateSchedule(
                    trainer_params.get("learning_rate_schedule", "linear")
                )
                h_size = int(trainer_params["hidden_units"])
                epsilon = float(trainer_params["epsilon"])
                beta = float(trainer_params["beta"])
                max_step = float(trainer_params["max_steps"])
                num_layers = int(trainer_params["num_layers"])
                vis_encode_type = EncoderType(
                    trainer_params.get("vis_encode_type", "simple")
                )
                self.burn_in_ratio = float(trainer_params.get("burn_in_ratio", 0.0))

                self.stream_names = list(self.reward_signals.keys())

                self.tf_optimizer: Optional[tf.train.AdamOptimizer] = None
                self.grads = None
                self.update_batch: Optional[tf.Operation] = None

                self.stats_name_to_update_name = {
                    "Losses/Value Loss": "value_loss",
                    "Losses/Policy Loss": "policy_loss",
                    "Policy/Learning Rate": "learning_rate",
                }
                if self.policy.use_recurrent:
                    self.m_size = self.policy.m_size
                    self.memory_in = tf.placeholder(
                        shape=[None, self.m_size],
                        dtype=tf.float32,
                        name="recurrent_value_in",
                    )

                if num_layers < 1:
                    num_layers = 1
                if policy.use_continuous_act:
                    self._create_cc_critic(h_size, num_layers, vis_encode_type)
                else:
                    self._create_dc_critic(h_size, num_layers, vis_encode_type)

                self.learning_rate = ModelUtils.create_learning_rate(
                    lr_schedule, lr, self.policy.global_step, int(max_step)
                )
                self._create_losses(
                    self.policy.total_log_probs,
                    self.old_log_probs,
                    self.value_heads,
                    self.policy.entropy,
                    beta,
                    epsilon,
                    lr,
                    max_step,
                )
                self._create_ppo_optimizer_ops()

            self.update_dict.update(
                {
                    "value_loss": self.value_loss,
                    "policy_loss": self.abs_policy_loss,
                    "update_batch": self.update_batch,
                    "learning_rate": self.learning_rate,
                }
            )

            self.policy.initialize_or_load()

    def _create_cc_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Continuous control critic (value) network.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: The type of visual encoder to use.
        """
        hidden_stream = ModelUtils.create_observation_streams(
            self.policy.visual_in,
            self.policy.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
        )[0]

        if self.policy.use_recurrent:
            hidden_value, memory_value_out = ModelUtils.create_recurrent_encoder(
                hidden_stream,
                self.memory_in,
                self.policy.sequence_length_ph,
                name="lstm_value",
            )
            self.memory_out = memory_value_out
        else:
            hidden_value = hidden_stream

        self.value_heads, self.value = ModelUtils.create_value_heads(
            self.stream_names, hidden_value
        )
        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )

        self.old_log_probs = tf.reduce_sum(
            (tf.identity(self.all_old_log_probs)), axis=1, keepdims=True
        )

    def _create_dc_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Discrete control critic (value) network.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: The type of visual encoder to use.
        """
        hidden_stream = ModelUtils.create_observation_streams(
            self.policy.visual_in,
            self.policy.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
        )[0]

        if self.policy.use_recurrent:
            hidden_value, memory_value_out = ModelUtils.create_recurrent_encoder(
                hidden_stream,
                self.memory_in,
                self.policy.sequence_length_ph,
                name="lstm_value",
            )
            self.memory_out = memory_value_out
        else:
            hidden_value = hidden_stream

        self.value_heads, self.value = ModelUtils.create_value_heads(
            self.stream_names, hidden_value
        )

        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )

        # Break old log probs into separate branches
        old_log_prob_branches = ModelUtils.break_into_branches(
            self.all_old_log_probs, self.policy.act_size
        )

        _, _, old_normalized_logits = ModelUtils.create_discrete_action_masking_layer(
            old_log_prob_branches, self.policy.action_masks, self.policy.act_size
        )

        action_idx = [0] + list(np.cumsum(self.policy.act_size))

        self.old_log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.policy.selected_actions[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                            logits=old_normalized_logits[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.policy.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )

    def _create_losses(
        self, probs, old_probs, value_heads, entropy, beta, epsilon, lr, max_step
    ):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value_heads: Value estimate tensors from each value stream
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """
        self.returns_holders = {}
        self.old_values = {}
        for name in value_heads.keys():
            returns_holder = tf.placeholder(
                shape=[None], dtype=tf.float32, name="{}_returns".format(name)
            )
            old_value = tf.placeholder(
                shape=[None], dtype=tf.float32, name="{}_value_estimate".format(name)
            )
            self.returns_holders[name] = returns_holder
            self.old_values[name] = old_value
        self.advantage = tf.placeholder(
            shape=[None], dtype=tf.float32, name="advantages"
        )
        advantage = tf.expand_dims(self.advantage, -1)

        decay_epsilon = tf.train.polynomial_decay(
            epsilon, self.policy.global_step, max_step, 0.1, power=1.0
        )
        decay_beta = tf.train.polynomial_decay(
            beta, self.policy.global_step, max_step, 1e-5, power=1.0
        )

        value_losses = []
        for name, head in value_heads.items():
            clipped_value_estimate = self.old_values[name] + tf.clip_by_value(
                tf.reduce_sum(head, axis=1) - self.old_values[name],
                -decay_epsilon,
                decay_epsilon,
            )
            v_opt_a = tf.squared_difference(
                self.returns_holders[name], tf.reduce_sum(head, axis=1)
            )
            v_opt_b = tf.squared_difference(
                self.returns_holders[name], clipped_value_estimate
            )
            value_loss = tf.reduce_mean(
                tf.dynamic_partition(tf.maximum(v_opt_a, v_opt_b), self.policy.mask, 2)[
                    1
                ]
            )
            value_losses.append(value_loss)
        self.value_loss = tf.reduce_mean(value_losses)

        r_theta = tf.exp(probs - old_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = (
            tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon)
            * advantage
        )
        self.policy_loss = -tf.reduce_mean(
            tf.dynamic_partition(tf.minimum(p_opt_a, p_opt_b), self.policy.mask, 2)[1]
        )
        # For cleaner stats reporting
        self.abs_policy_loss = tf.abs(self.policy_loss)

        self.loss = (
            self.policy_loss
            + 0.5 * self.value_loss
            - decay_beta
            * tf.reduce_mean(tf.dynamic_partition(entropy, self.policy.mask, 2)[1])
        )

    def _create_ppo_optimizer_ops(self):
        self.tf_optimizer = self.create_optimizer_op(self.learning_rate)
        self.grads = self.tf_optimizer.compute_gradients(self.loss)
        self.update_batch = self.tf_optimizer.minimize(self.loss)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        feed_dict = self._construct_feed_dict(batch, num_sequences)
        stats_needed = self.stats_name_to_update_name
        update_stats = {}
        # Collect feed dicts for all reward signals.
        for _, reward_signal in self.reward_signals.items():
            feed_dict.update(
                reward_signal.prepare_update(self.policy, batch, num_sequences)
            )
            stats_needed.update(reward_signal.stats_name_to_update_name)

        update_vals = self._execute_model(feed_dict, self.update_dict)
        for stat_name, update_name in stats_needed.items():
            update_stats[stat_name] = update_vals[update_name]
        return update_stats

    def _construct_feed_dict(
        self, mini_batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        # Do an optional burn-in for memories
        num_burn_in = int(self.burn_in_ratio * self.policy.sequence_length)
        burn_in_mask = np.ones((self.policy.sequence_length), dtype=np.float32)
        burn_in_mask[range(0, num_burn_in)] = 0
        burn_in_mask = np.tile(burn_in_mask, num_sequences)
        feed_dict = {
            self.policy.batch_size_ph: num_sequences,
            self.policy.sequence_length_ph: self.policy.sequence_length,
            self.policy.mask_input: mini_batch["masks"] * burn_in_mask,
            self.advantage: mini_batch["advantages"],
            self.all_old_log_probs: mini_batch["action_probs"],
        }
        for name in self.reward_signals:
            feed_dict[self.returns_holders[name]] = mini_batch[
                "{}_returns".format(name)
            ]
            feed_dict[self.old_values[name]] = mini_batch[
                "{}_value_estimates".format(name)
            ]

        if self.policy.output_pre is not None and "actions_pre" in mini_batch:
            feed_dict[self.policy.output_pre] = mini_batch["actions_pre"]
        else:
            feed_dict[self.policy.output] = mini_batch["actions"]
            if self.policy.use_recurrent:
                feed_dict[self.policy.prev_action] = mini_batch["prev_action"]
            feed_dict[self.policy.action_masks] = mini_batch["action_mask"]
        if "vector_obs" in mini_batch:
            feed_dict[self.policy.vector_in] = mini_batch["vector_obs"]
        if self.policy.vis_obs_size > 0:
            for i, _ in enumerate(self.policy.visual_in):
                feed_dict[self.policy.visual_in[i]] = mini_batch["visual_obs%d" % i]
        if self.policy.use_recurrent:
            feed_dict[self.policy.memory_in] = [
                mini_batch["memory"][i]
                for i in range(
                    0, len(mini_batch["memory"]), self.policy.sequence_length
                )
            ]
            feed_dict[self.memory_in] = self._make_zero_mem(
                self.m_size, mini_batch.num_experiences
            )
        return feed_dict

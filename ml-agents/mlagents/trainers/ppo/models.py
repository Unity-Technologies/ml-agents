import logging
import numpy as np

import tensorflow as tf
from mlagents.trainers.models import LearningModel, EncoderType, LearningRateSchedule

logger = logging.getLogger("mlagents.trainers")


class PPOModel(LearningModel):
    def __init__(
        self,
        brain,
        lr=1e-4,
        lr_schedule=LearningRateSchedule.LINEAR,
        h_size=128,
        epsilon=0.2,
        beta=1e-3,
        max_step=5e6,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        m_size=None,
        seed=0,
        stream_names=None,
        vis_encode_type=EncoderType.SIMPLE,
    ):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: BrainInfo used to generate specific network graph.
        :param lr: Learning rate.
        :param lr_schedule: Learning rate decay schedule.
        :param h_size: Size of hidden layers
        :param epsilon: Value for policy-divergence threshold.
        :param beta: Strength of entropy regularization.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers Number of hidden layers between encoded input and policy & value layers
        :param m_size: Size of brain memory.
        :param seed: Seed to use for initialization of model.
        :param stream_names: List of names of value streams. Usually, a list of the Reward Signals being used.
        :return: a sub-class of PPOAgent tailored to the environment.
        """
        LearningModel.__init__(
            self, m_size, normalize, use_recurrent, brain, seed, stream_names
        )
        if num_layers < 1:
            num_layers = 1
        if brain.vector_action_space_type == "continuous":
            self.create_cc_actor_critic(h_size, num_layers, vis_encode_type)
            self.entropy = tf.ones_like(tf.reshape(self.value, [-1])) * self.entropy
        else:
            self.create_dc_actor_critic(h_size, num_layers, vis_encode_type)
        self.learning_rate = self.create_learning_rate(
            lr_schedule, lr, self.global_step, max_step
        )
        self.create_losses(
            self.log_probs,
            self.old_log_probs,
            self.value_heads,
            self.entropy,
            beta,
            epsilon,
            lr,
            max_step,
        )

    def create_cc_actor_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_streams = self.create_observation_streams(
            2, h_size, num_layers, vis_encode_type
        )

        if self.use_recurrent:
            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            _half_point = int(self.m_size / 2)
            hidden_policy, memory_policy_out = self.create_recurrent_encoder(
                hidden_streams[0],
                self.memory_in[:, :_half_point],
                self.sequence_length,
                name="lstm_policy",
            )

            hidden_value, memory_value_out = self.create_recurrent_encoder(
                hidden_streams[1],
                self.memory_in[:, _half_point:],
                self.sequence_length,
                name="lstm_value",
            )
            self.memory_out = tf.concat(
                [memory_policy_out, memory_value_out], axis=1, name="recurrent_out"
            )
        else:
            hidden_policy = hidden_streams[0]
            hidden_value = hidden_streams[1]

        mu = tf.layers.dense(
            hidden_policy,
            self.act_size[0],
            activation=None,
            kernel_initializer=LearningModel.scaled_init(0.01),
            reuse=tf.AUTO_REUSE,
        )

        self.log_sigma_sq = tf.get_variable(
            "log_sigma_squared",
            [self.act_size[0]],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
        )

        sigma_sq = tf.exp(self.log_sigma_sq)

        self.epsilon = tf.placeholder(
            shape=[None, self.act_size[0]], dtype=tf.float32, name="epsilon"
        )
        # Clip and scale output to ensure actions are always within [-1, 1] range.
        self.output_pre = mu + tf.sqrt(sigma_sq) * self.epsilon
        output_post = tf.clip_by_value(self.output_pre, -3, 3) / 3
        self.output = tf.identity(output_post, name="action")
        self.selected_actions = tf.stop_gradient(output_post)

        # Compute probability of model output.
        all_probs = (
            -0.5 * tf.square(tf.stop_gradient(self.output_pre) - mu) / sigma_sq
            - 0.5 * tf.log(2.0 * np.pi)
            - 0.5 * self.log_sigma_sq
        )

        self.all_log_probs = tf.identity(all_probs, name="action_probs")

        self.entropy = 0.5 * tf.reduce_mean(
            tf.log(2 * np.pi * np.e) + self.log_sigma_sq
        )

        self.create_value_heads(self.stream_names, hidden_value)

        self.all_old_log_probs = tf.placeholder(
            shape=[None, self.act_size[0]], dtype=tf.float32, name="old_probabilities"
        )

        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.log_probs = tf.reduce_sum(
            (tf.identity(self.all_log_probs)), axis=1, keepdims=True
        )
        self.old_log_probs = tf.reduce_sum(
            (tf.identity(self.all_old_log_probs)), axis=1, keepdims=True
        )

    def create_dc_actor_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_streams = self.create_observation_streams(
            1, h_size, num_layers, vis_encode_type
        )
        hidden = hidden_streams[0]

        if self.use_recurrent:
            self.prev_action = tf.placeholder(
                shape=[None, len(self.act_size)], dtype=tf.int32, name="prev_action"
            )
            prev_action_oh = tf.concat(
                [
                    tf.one_hot(self.prev_action[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            hidden = tf.concat([hidden, prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            hidden, memory_out = self.create_recurrent_encoder(
                hidden, self.memory_in, self.sequence_length
            )
            self.memory_out = tf.identity(memory_out, name="recurrent_out")

        policy_branches = []
        for size in self.act_size:
            policy_branches.append(
                tf.layers.dense(
                    hidden,
                    size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=LearningModel.scaled_init(0.01),
                )
            )

        self.all_log_probs = tf.concat(
            [branch for branch in policy_branches], axis=1, name="action_probs"
        )

        self.action_masks = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
        )
        output, _, normalized_logits = self.create_discrete_action_masking_layer(
            self.all_log_probs, self.action_masks, self.act_size
        )

        self.output = tf.identity(output)
        self.normalized_logits = tf.identity(normalized_logits, name="action")

        self.create_value_heads(self.stream_names, hidden)

        self.action_holder = tf.placeholder(
            shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder"
        )
        self.action_oh = tf.concat(
            [
                tf.one_hot(self.action_holder[:, i], self.act_size[i])
                for i in range(len(self.act_size))
            ],
            axis=1,
        )
        self.selected_actions = tf.stop_gradient(self.action_oh)

        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="old_probabilities"
        )
        _, _, old_normalized_logits = self.create_discrete_action_masking_layer(
            self.all_old_log_probs, self.action_masks, self.act_size
        )

        action_idx = [0] + list(np.cumsum(self.act_size))

        self.entropy = tf.reduce_sum(
            (
                tf.stack(
                    [
                        tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=tf.nn.softmax(
                                self.all_log_probs[:, action_idx[i] : action_idx[i + 1]]
                            ),
                            logits=self.all_log_probs[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
        )

        self.log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.action_oh[:, action_idx[i] : action_idx[i + 1]],
                            logits=normalized_logits[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )
        self.old_log_probs = tf.reduce_sum(
            (
                tf.stack(
                    [
                        -tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.action_oh[:, action_idx[i] : action_idx[i + 1]],
                            logits=old_normalized_logits[
                                :, action_idx[i] : action_idx[i + 1]
                            ],
                        )
                        for i in range(len(self.act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )

    def create_losses(
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
            epsilon, self.global_step, max_step, 0.1, power=1.0
        )
        decay_beta = tf.train.polynomial_decay(
            beta, self.global_step, max_step, 1e-5, power=1.0
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
                tf.dynamic_partition(tf.maximum(v_opt_a, v_opt_b), self.mask, 2)[1]
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
            tf.dynamic_partition(tf.minimum(p_opt_a, p_opt_b), self.mask, 2)[1]
        )
        # For cleaner stats reporting
        self.abs_policy_loss = tf.abs(self.policy_loss)

        self.loss = (
            self.policy_loss
            + 0.5 * self.value_loss
            - decay_beta
            * tf.reduce_mean(tf.dynamic_partition(entropy, self.mask, 2)[1])
        )

    def create_ppo_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.grads = self.optimizer.compute_gradients(self.loss)
        self.update_batch = self.optimizer.minimize(self.loss)

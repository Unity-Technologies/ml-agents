import logging
from typing import Optional, Any, Dict

import numpy as np
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed
from mlagents.trainers.models import LearningModel, EncoderType, LearningRateSchedule
from mlagents.trainers.optimizer import TFOptimizer
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.ppo.models import PPOModel


logger = logging.getLogger("mlagents.trainers")


class PPOOptimizer(TFOptimizer):
    def __init__(
        self,
        brain,
        sess,
        policy,
        reward_signal_configs,
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
        vis_encode_type=EncoderType.SIMPLE,
    ):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: brain parameters used to generate specific network graph.
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
        super().__init__(sess, policy, reward_signal_configs)

        self.stream_names = self.reward_signals.keys()

        self.optimizer: Optional[tf.train.AdamOptimizer] = None
        self.grads = None
        self.update_batch: Optional[tf.Operation] = None

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        if num_layers < 1:
            num_layers = 1
        if brain.vector_action_space_type == "continuous":
            self.create_cc_critic(h_size, num_layers, vis_encode_type)
        else:
            self.create_dc_actor_critic(h_size, num_layers, vis_encode_type)

        self.learning_rate = LearningModel.create_learning_rate(
            lr_schedule, lr, self.policy.global_step, max_step
        )
        self.create_losses(
            self.policy.log_probs,
            self.old_log_probs,
            self.value_heads,
            self.policy.entropy,
            beta,
            epsilon,
            lr,
            max_step,
        )

    def create_cc_critic(
        self, h_size: int, num_layers: int, vis_encode_type: EncoderType
    ) -> None:
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_stream = LearningModel.create_observation_streams(
            self.policy.visual_in,
            self.policy.processed_vector_in,
            1,
            h_size,
            num_layers,
            vis_encode_type,
            stream_scopes=["optimizer"],
        )[0]

        if self.policy.use_recurrent:
            self.memory_in = tf.placeholder(
                shape=[None, self.policy.m_size], dtype=tf.float32, name="recurrent_in"
            )
            _half_point = int(self.policy.m_size / 2)

            hidden_value, memory_value_out = LearningModel.create_recurrent_encoder(
                hidden_stream,
                self.memory_in[:, _half_point:],
                self.policy.sequence_length,
                name="lstm_value",
            )
            self.memory_out = memory_value_out
        else:
            hidden_value = hidden_stream

        self.create_value_heads(self.stream_names, hidden_value)
        self.all_old_log_probs = tf.placeholder(
            shape=[None, self.policy.act_size[0]],
            dtype=tf.float32,
            name="old_probabilities",
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
        hidden_streams = LearningModel.create_observation_streams(
            self.policy.visual_in,
            self.policy.processed_vector_in,
            2,
            h_size,
            num_layers,
            vis_encode_type,
        )

        if self.policy.use_recurrent:
            self.prev_action = tf.placeholder(
                shape=[None, len(self.policy.act_size)],
                dtype=tf.int32,
                name="prev_action",
            )
            prev_action_oh = tf.concat(
                [
                    tf.one_hot(self.prev_action[:, i], self.policy.act_size[i])
                    for i in range(len(self.policy.act_size))
                ],
                axis=1,
            )
            hidden_policy = tf.concat([hidden_streams[0], prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(
                shape=[None, self.policy.m_size], dtype=tf.float32, name="recurrent_in"
            )
            _half_point = int(self.policy.m_size / 2)
            hidden_policy, memory_policy_out = LearningModel.create_recurrent_encoder(
                hidden_policy,
                self.memory_in[:, :_half_point],
                self.policy.sequence_length,
                name="lstm_policy",
            )

            hidden_value, memory_value_out = LearningModel.create_recurrent_encoder(
                hidden_streams[1],
                self.memory_in[:, _half_point:],
                self.policy.sequence_length,
                name="lstm_value",
            )
            self.memory_out = tf.concat(
                [memory_policy_out, memory_value_out], axis=1, name="recurrent_out"
            )
        else:
            hidden_policy = hidden_streams[0]
            hidden_value = hidden_streams[1]

        policy_branches = []
        for size in self.policy.act_size:
            policy_branches.append(
                tf.layers.dense(
                    hidden_policy,
                    size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=LearningModel.scaled_init(0.01),
                )
            )

        self.all_log_probs = tf.concat(policy_branches, axis=1, name="action_probs")

        self.action_masks = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="action_masks",
        )
        output, _, normalized_logits = LearningModel.create_discrete_action_masking_layer(
            self.all_log_probs, self.action_masks, self.policy.act_size
        )

        self.output = tf.identity(output)
        self.normalized_logits = tf.identity(normalized_logits, name="action")

        self.create_value_heads(self.stream_names, hidden_value)

        self.action_holder = tf.placeholder(
            shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder"
        )
        self.action_oh = tf.concat(
            [
                tf.one_hot(self.action_holder[:, i], self.policy.act_size[i])
                for i in range(len(self.policy.act_size))
            ],
            axis=1,
        )
        self.selected_actions = tf.stop_gradient(self.action_oh)

        self.all_old_log_probs = tf.placeholder(
            shape=[None, sum(self.policy.act_size)],
            dtype=tf.float32,
            name="old_probabilities",
        )
        _, _, old_normalized_logits = LearningModel.create_discrete_action_masking_layer(
            self.all_old_log_probs, self.action_masks, self.policy.act_size
        )

        action_idx = [0] + list(np.cumsum(self.policy.act_size))

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
                        for i in range(len(self.policy.act_size))
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
                        for i in range(len(self.policy.act_size))
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
                        for i in range(len(self.policy.act_size))
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

    def create_ppo_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.grads = self.optimizer.compute_gradients(self.loss)
        self.update_batch = self.optimizer.minimize(self.loss)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        feed_dict = self.construct_feed_dict(self.policy, batch, num_sequences)
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

    def construct_feed_dict(
        self, model: PPOModel, mini_batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        feed_dict = {
            model.batch_size: num_sequences,
            model.sequence_length: model.sequence_length,
            model.mask_input: mini_batch["masks"],
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

        if "actions_pre" in mini_batch:
            feed_dict[model.output_pre] = mini_batch["actions_pre"]
        else:
            feed_dict[model.action_holder] = mini_batch["actions"]
            if "prev_action" in mini_batch:
                feed_dict[model.prev_action] = mini_batch["prev_action"]
            feed_dict[model.action_masks] = mini_batch["action_mask"]
        if "vector_obs" in mini_batch:
            feed_dict[model.vector_in] = mini_batch["vector_obs"]
        if model.vis_obs_size > 0:
            for i, _ in enumerate(model.visual_in):
                feed_dict[model.visual_in[i]] = mini_batch["visual_obs%d" % i]
        if "memory" in mini_batch:
            mem_in = [
                mini_batch["memory"][i]
                for i in range(
                    0, len(mini_batch["memory"]), self.policy.sequence_length
                )
            ]
            feed_dict[model.memory_in] = mem_in
        return feed_dict

    def _execute_model(self, feed_dict, out_dict):
        """
        Executes model.
        :param feed_dict: Input dictionary mapping nodes to input data.
        :param out_dict: Output dictionary mapping names to nodes.
        :return: Dictionary mapping names to input data.
        """
        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

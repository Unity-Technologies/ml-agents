import logging
import numpy as np

import tensorflow as tf
from mlagents.trainers.models import LearningModel

logger = logging.getLogger("mlagents.trainers")


class PPOModel(LearningModel):
    def __init__(
        self,
        brain,
        lr=1e-4,
        h_size=128,
        epsilon=0.2,
        beta=1e-3,
        max_step=5e6,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        m_size=None,
        use_curiosity=False,
        curiosity_strength=0.01,
        curiosity_enc_size=128,
        seed=0,
    ):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: BrainInfo used to generate specific network graph.
        :param lr: Learning rate.
        :param h_size: Size of hidden layers
        :param epsilon: Value for policy-divergence threshold.
        :param beta: Strength of entropy regularization.
        :return: a sub-class of PPOAgent tailored to the environment.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers Number of hidden layers between encoded input and policy & value layers
        :param m_size: Size of brain memory.
        """
        LearningModel.__init__(self, m_size, normalize, use_recurrent, brain, seed)
        self.use_curiosity = use_curiosity
        if num_layers < 1:
            num_layers = 1
        self.last_reward, self.new_reward, self.update_reward = (
            self.create_reward_encoder()
        )
        if brain.vector_action_space_type == "continuous":
            self.create_cc_actor_critic(h_size, num_layers)
            self.entropy = tf.ones_like(tf.reshape(self.value, [-1])) * self.entropy
        else:
            self.create_dc_actor_critic(h_size, num_layers)
        if self.use_curiosity:
            self.curiosity_enc_size = curiosity_enc_size
            self.curiosity_strength = curiosity_strength
            encoded_state, encoded_next_state = self.create_curiosity_encoders()
            self.create_inverse_model(encoded_state, encoded_next_state)
            self.create_forward_model(encoded_state, encoded_next_state)
        self.create_ppo_optimizer(
            self.log_probs,
            self.old_log_probs,
            self.value,
            self.entropy,
            beta,
            epsilon,
            lr,
            max_step,
        )

    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(
            0, name="last_reward", trainable=False, dtype=tf.float32
        )
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name="new_reward")
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    def create_curiosity_encoders(self):
        """
        Creates state encoders for current and future observations.
        Used for implementation of ﻿Curiosity-driven Exploration by Self-supervised Prediction
        See https://arxiv.org/abs/1705.05363 for more details.
        :return: current and future state encoder tensors.
        """
        encoded_state_list = []
        encoded_next_state_list = []

        if self.vis_obs_size > 0:
            self.next_visual_in = []
            visual_encoders = []
            next_visual_encoders = []
            for i in range(self.vis_obs_size):
                # Create input ops for next (t+1) visual observations.
                next_visual_input = self.create_visual_input(
                    self.brain.camera_resolutions[i],
                    name="next_visual_observation_" + str(i),
                )
                self.next_visual_in.append(next_visual_input)

                # Create the encoder ops for current and next visual input. Not that these encoders are siamese.
                encoded_visual = self.create_visual_observation_encoder(
                    self.visual_in[i],
                    self.curiosity_enc_size,
                    self.swish,
                    1,
                    "stream_{}_visual_obs_encoder".format(i),
                    False,
                )

                encoded_next_visual = self.create_visual_observation_encoder(
                    self.next_visual_in[i],
                    self.curiosity_enc_size,
                    self.swish,
                    1,
                    "stream_{}_visual_obs_encoder".format(i),
                    True,
                )
                visual_encoders.append(encoded_visual)
                next_visual_encoders.append(encoded_next_visual)

            hidden_visual = tf.concat(visual_encoders, axis=1)
            hidden_next_visual = tf.concat(next_visual_encoders, axis=1)
            encoded_state_list.append(hidden_visual)
            encoded_next_state_list.append(hidden_next_visual)

        if self.vec_obs_size > 0:
            # Create the encoder ops for current and next vector input. Not that these encoders are siamese.
            # Create input op for next (t+1) vector observation.
            self.next_vector_in = tf.placeholder(
                shape=[None, self.vec_obs_size],
                dtype=tf.float32,
                name="next_vector_observation",
            )

            encoded_vector_obs = self.create_vector_observation_encoder(
                self.vector_in,
                self.curiosity_enc_size,
                self.swish,
                2,
                "vector_obs_encoder",
                False,
            )
            encoded_next_vector_obs = self.create_vector_observation_encoder(
                self.next_vector_in,
                self.curiosity_enc_size,
                self.swish,
                2,
                "vector_obs_encoder",
                True,
            )
            encoded_state_list.append(encoded_vector_obs)
            encoded_next_state_list.append(encoded_next_vector_obs)

        encoded_state = tf.concat(encoded_state_list, axis=1)
        encoded_next_state = tf.concat(encoded_next_state_list, axis=1)
        return encoded_state, encoded_next_state

    def create_inverse_model(self, encoded_state, encoded_next_state):
        """
        Creates inverse model TensorFlow ops for Curiosity module.
        Predicts action taken given current and future encoded states.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        combined_input = tf.concat([encoded_state, encoded_next_state], axis=1)
        hidden = tf.layers.dense(combined_input, 256, activation=self.swish)
        if self.brain.vector_action_space_type == "continuous":
            pred_action = tf.layers.dense(hidden, self.act_size[0], activation=None)
            squared_difference = tf.reduce_sum(
                tf.squared_difference(pred_action, self.selected_actions), axis=1
            )
            self.inverse_loss = tf.reduce_mean(
                tf.dynamic_partition(squared_difference, self.mask, 2)[1]
            )
        else:
            pred_action = tf.concat(
                [
                    tf.layers.dense(hidden, self.act_size[i], activation=tf.nn.softmax)
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            cross_entropy = tf.reduce_sum(
                -tf.log(pred_action + 1e-10) * self.selected_actions, axis=1
            )
            self.inverse_loss = tf.reduce_mean(
                tf.dynamic_partition(cross_entropy, self.mask, 2)[1]
            )

    def create_forward_model(self, encoded_state, encoded_next_state):
        """
        Creates forward model TensorFlow ops for Curiosity module.
        Predicts encoded future state based on encoded current state and given action.
        :param encoded_state: Tensor corresponding to encoded current state.
        :param encoded_next_state: Tensor corresponding to encoded next state.
        """
        combined_input = tf.concat([encoded_state, self.selected_actions], axis=1)
        hidden = tf.layers.dense(combined_input, 256, activation=self.swish)
        # We compare against the concatenation of all observation streams, hence `self.vis_obs_size + int(self.vec_obs_size > 0)`.
        pred_next_state = tf.layers.dense(
            hidden,
            self.curiosity_enc_size * (self.vis_obs_size + int(self.vec_obs_size > 0)),
            activation=None,
        )

        squared_difference = 0.5 * tf.reduce_sum(
            tf.squared_difference(pred_next_state, encoded_next_state), axis=1
        )
        self.intrinsic_reward = tf.clip_by_value(
            self.curiosity_strength * squared_difference, 0, 1
        )
        self.forward_loss = tf.reduce_mean(
            tf.dynamic_partition(squared_difference, self.mask, 2)[1]
        )

    def create_ppo_optimizer(
        self, probs, old_probs, value, entropy, beta, epsilon, lr, max_step
    ):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value: Current value estimate
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """
        self.returns_holder = tf.placeholder(
            shape=[None], dtype=tf.float32, name="discounted_rewards"
        )
        self.advantage = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name="advantages"
        )
        self.learning_rate = tf.train.polynomial_decay(
            lr, self.global_step, max_step, 1e-10, power=1.0
        )

        self.old_value = tf.placeholder(
            shape=[None], dtype=tf.float32, name="old_value_estimates"
        )

        decay_epsilon = tf.train.polynomial_decay(
            epsilon, self.global_step, max_step, 0.1, power=1.0
        )
        decay_beta = tf.train.polynomial_decay(
            beta, self.global_step, max_step, 1e-5, power=1.0
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        clipped_value_estimate = self.old_value + tf.clip_by_value(
            tf.reduce_sum(value, axis=1) - self.old_value, -decay_epsilon, decay_epsilon
        )

        v_opt_a = tf.squared_difference(
            self.returns_holder, tf.reduce_sum(value, axis=1)
        )
        v_opt_b = tf.squared_difference(self.returns_holder, clipped_value_estimate)
        self.value_loss = tf.reduce_mean(
            tf.dynamic_partition(tf.maximum(v_opt_a, v_opt_b), self.mask, 2)[1]
        )

        # Here we calculate PPO policy loss. In continuous control this is done independently for each action gaussian
        # and then averaged together. This provides significantly better performance than treating the probability
        # as an average of probabilities, or as a joint probability.
        r_theta = tf.exp(probs - old_probs)
        p_opt_a = r_theta * self.advantage
        p_opt_b = (
            tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon)
            * self.advantage
        )
        self.policy_loss = -tf.reduce_mean(
            tf.dynamic_partition(tf.minimum(p_opt_a, p_opt_b), self.mask, 2)[1]
        )

        self.loss = (
            self.policy_loss
            + 0.5 * self.value_loss
            - decay_beta
            * tf.reduce_mean(tf.dynamic_partition(entropy, self.mask, 2)[1])
        )

        if self.use_curiosity:
            self.loss += 10 * (0.2 * self.forward_loss + 0.8 * self.inverse_loss)
        self.update_batch = optimizer.minimize(self.loss)

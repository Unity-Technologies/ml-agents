import logging

import tensorflow as tf
from unitytrainers.models import LearningModel

logger = logging.getLogger("unityagents")


class PPOCurioModel(LearningModel):
    def __init__(self, brain, lr=1e-4, h_size=128, epsilon=0.2, beta=1e-3, max_step=5e6,
                 normalize=False, use_recurrent=False, num_layers=2, m_size=None):
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
        LearningModel.__init__(self, m_size, normalize, use_recurrent, brain)
        if num_layers < 1:
            num_layers = 1
        self.last_reward, self.new_reward, self.update_reward = self.create_reward_encoder()
        if brain.vector_action_space_type == "continuous":
            self.create_cc_actor_critic(h_size, num_layers)
            self.entropy = tf.ones_like(tf.reshape(self.value, [-1])) * self.entropy
        else:
            self.create_dc_actor_critic(h_size, num_layers)
            s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
            a_size = brain.vector_action_space_size
            encoded_state, encoded_next_state = self.create_inverse_model(a_size, s_size)
            self.create_forward_model(encoded_state, encoded_next_state)
        self.create_ppo_optimizer(self.probs, self.old_probs, self.value,
                                  self.entropy, beta, epsilon, lr, max_step)

    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    @staticmethod
    def encode_state(state, reuse):
        with tf.name_scope("state_encoder"):
            hidden_1 = tf.layers.dense(state, 128, reuse=reuse, activation=tf.nn.elu, name="encode_1")
            encoded_state = tf.layers.dense(hidden_1, 128, reuse=reuse, activation=tf.nn.elu, name="encode_2")
            return encoded_state

    def create_inverse_model(self, a_size, s_size):
        self.next_state = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='next_vector_observation')
        encoded_state = self.encode_state(self.vector_in, reuse=False)
        encoded_next_state = self.encode_state(self.next_state, reuse=True)

        combined = tf.concat([encoded_state, encoded_next_state], axis=1)
        pred_action = tf.layers.dense(combined, a_size, activation=tf.nn.sigmoid)
        self.inverse_loss = tf.reduce_sum(-tf.log(pred_action + 1e-10) * self.selected_actions)
        return encoded_state, encoded_next_state

    def create_forward_model(self, encoded_state, encoded_next_state):
        combined = tf.concat([encoded_state, self.selected_actions], axis=1)
        hidden = tf.layers.dense(combined, 128, activation=tf.nn.elu)
        pred_next_state = tf.layers.dense(hidden, 128, activation=None)

        forward_distance = tf.squared_difference(pred_next_state, encoded_next_state)
        self.intrinsic_reward = tf.reduce_mean(forward_distance, axis=1)
        self.forward_loss = tf.reduce_sum(forward_distance)

    def create_ppo_optimizer(self, probs, old_probs, value, entropy, beta, epsilon, lr, max_step):
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

        self.returns_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='discounted_rewards')
        self.advantage = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')
        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step, max_step, 1e-10, power=1.0)

        self.old_value = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')
        self.mask_input = tf.placeholder(shape=[None], dtype=tf.float32, name='masks')

        decay_epsilon = tf.train.polynomial_decay(epsilon, self.global_step, max_step, 0.1, power=1.0)
        decay_beta = tf.train.polynomial_decay(beta, self.global_step, max_step, 1e-5, power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.mask = tf.equal(self.mask_input, 1.0)

        clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(value, axis=1) - self.old_value,
                                                                   - decay_epsilon, decay_epsilon)

        v_opt_a = tf.squared_difference(self.returns_holder, tf.reduce_sum(value, axis=1))
        v_opt_b = tf.squared_difference(self.returns_holder, clipped_value_estimate)
        self.value_loss = tf.reduce_mean(tf.boolean_mask(tf.maximum(v_opt_a, v_opt_b), self.mask))

        self.r_theta = probs / (old_probs + 1e-10)
        self.p_opt_a = self.r_theta * self.advantage
        self.p_opt_b = tf.clip_by_value(self.r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.boolean_mask(tf.minimum(self.p_opt_a, self.p_opt_b), self.mask))
        self.loss = self.policy_loss + 0.5 * self.value_loss - decay_beta * tf.reduce_mean(
            tf.boolean_mask(entropy, self.mask)) + self.forward_loss + self.inverse_loss
        self.update_batch = optimizer.minimize(self.loss)

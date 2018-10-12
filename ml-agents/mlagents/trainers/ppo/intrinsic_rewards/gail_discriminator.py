import tensorflow as tf
import numpy as np

class Discriminator(object):
    def __init__(self, policy_model, h_size, lr):
        self.h_size = h_size
        self.policy_model = policy_model

        self.make_inputs()
        self.create_network()
        self.create_loss(lr)

    def make_inputs(self):
        self.obs_in_expert = tf.placeholder(
            shape=[None, self.policy_model.vec_obs_size], dtype=tf.float32)
        self.action_in_expert = tf.placeholder(
            shape=[None, np.cumsum(self.policy_model.act_size)], dtype=tf.float32)

    def create_encoder(self, state_in, action_in, reuse):
        with tf.variable_scope("discriminator"):
            concat_input = tf.concat([state_in, action_in], axis=1)

            hidden_1 = tf.layers.dense(
                concat_input, self.h_size, activation=tf.nn.elu,
                name="d_hidden_1", reuse=reuse)

            hidden_2 = tf.layers.dense(
                hidden_1, self.h_size, activation=tf.nn.elu,
                name="d_hidden_2", reuse=reuse)

            estimate = tf.layers.dense(hidden_2, 1, activation=tf.nn.sigmoid,
                                       name="d_estimate", reuse=reuse)
            return estimate

    def create_network(self):
        self.expert_estimate = self.create_encoder(self.obs_in_expert, self.action_in_expert, False)
        self.policy_estimate = self.create_encoder(self.policy_model.vector_in,
                                                   self.policy_model.selected_actions, True)
        self.intrinsic_reward = tf.reshape(self.policy_estimate, [-1], name="GAIL_reward")

    def create_loss(self, learning_rate):
        self.mean_expert_estimate = tf.reduce_mean(self.expert_estimate)
        self.mean_policy_estimate = tf.reduce_mean(self.policy_estimate)
        self.loss = -tf.reduce_mean(
            tf.log(self.expert_estimate + 1e-10) + tf.log(1.0 - self.policy_estimate + 1e-10))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

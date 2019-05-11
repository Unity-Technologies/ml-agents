import tensorflow as tf
import numpy as np
from mlagents.trainers.models import LearningModel


class BCModel(object):
    def __init__(self, policy_model: LearningModel, lr, anneal_steps):
        """
        The GAIL reward generator.
        https://arxiv.org/abs/1606.03476
        :param policy_model: The policy of the learning algorithm
        :param h_size: Size of the hidden layer for the discriminator
        :param lr: The learning Rate for the discriminator
        :param encoding_size: The encoding size for the encoder
        """
        self.z_size = 128
        self.alpha = 0.0005
        self.mutual_information = 0.5
        self.policy_model = policy_model
        self.expert_visual_in = self.policy_model.visual_in
        self.obs_in_expert = self.policy_model.vector_in
        # self.make_beta()
        self.make_inputs()
        # self.create_network()
        self.create_loss(lr, anneal_steps)

    def make_inputs(self):
        """
        Creates the input layers for the discriminator
        """
        self.done_expert = tf.placeholder(
            shape=[None, 1], dtype=tf.float32)
        self.done_policy = tf.placeholder(
            shape=[None, 1], dtype=tf.float32)

        if self.policy_model.brain.vector_action_space_type == 'continuous':
            action_length = self.policy_model.act_size[0]
            self.action_in_expert = tf.placeholder(
                shape=[None, action_length], dtype=tf.float32)
            self.expert_action = tf.identity(self.action_in_expert)
        else:
            action_length = len(self.policy_model.act_size)
            self.action_in_expert = tf.placeholder(
                shape=[None, action_length], dtype=tf.int32)
            self.expert_action = tf.concat([
                tf.one_hot(self.action_in_expert[:, i], self.policy_model.act_size[i]) for i in
                range(len(self.policy_model.act_size))], axis=1)

    def create_loss(self, learning_rate, anneal_steps):
        """
        Creates the loss and update nodes for the GAIL reward generator
        :param learning_rate: The learning rate for the optimizer
        """
        selected_action = self.policy_model.output
        action_size = self.policy_model.act_size
        if self.policy_model.brain.vector_action_space_type == "continuous":
            # self.expert_action = tf.placeholder(shape=[None, action_size[0]],
            #                                     dtype=tf.float32,
            #                                     name="teacher_action")
            entropy = 0.5 * tf.reduce_mean(tf.log(2 * np.pi * np.e) + self.policy_model.log_sigma_sq)
            self.loss = tf.reduce_mean(tf.squared_difference(selected_action, self.expert_action)) #/ self.policy_model.n_sequences \
                        #- tf.reduce_mean(entropy)
        else:
            log_probs = self.policy_model.all_log_probs
            # self.expert_action = tf.placeholder(shape=[None, len(action_size)], dtype=tf.int32)
            action_idx = [0] + list(np.cumsum(action_size))
            entropy = tf.reduce_sum((tf.stack([
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.nn.softmax(log_probs[:, action_idx[i]:action_idx[i + 1]]),
                    logits=log_probs[:, action_idx[i]:action_idx[i + 1]])
                for i in range(len(action_size))], axis=1)), axis=1)
            self.loss = tf.reduce_mean(-tf.log(tf.nn.softmax(log_probs)+ 0*1e-10) * self.expert_action) #- 1 * tf.reduce_sum(entropy)

        #self.loss = tf.train.polynomial_decay(self.loss, self.policy_model.global_step, max_step*decay_duration, 0.0, power=1.0)
        if anneal_steps > 0:
            annealed_learning_rate = tf.train.polynomial_decay(learning_rate, self.policy_model.global_step, anneal_steps, 0.0, power=1.0)
        else:
            annealed_learning_rate = learning_rate

        optimizer = tf.train.AdamOptimizer(learning_rate=annealed_learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

import tensorflow as tf
import numpy as np
from mlagents.trainers.models import LearningModel


class BCModel(object):
    def __init__(self, policy_model: LearningModel, h_size, lr, encoding_size, max_step, decay_duration):
        """
        The GAIL reward generator.
        https://arxiv.org/abs/1606.03476
        :param policy_model: The policy of the learning algorithm
        :param h_size: Size of the hidden layer for the discriminator
        :param lr: The learning Rate for the discriminator
        :param encoding_size: The encoding size for the encoder
        """
        self.h_size = h_size
        self.z_size = 128
        self.alpha = 0.0005
        self.mutual_information = 0.5
        self.policy_model = policy_model
        self.encoding_size = encoding_size
        self.use_vail = True
        self.use_actions = False#True # Not using actions
        self.expert_visual_in = self.policy_model.visual_in
        self.obs_in_expert = self.policy_model.vector_in
        # self.make_beta()
        self.make_inputs()
        # self.create_network()
        self.create_loss(lr, max_step, decay_duration)

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

        encoded_policy_list = []
        encoded_expert_list = []

        if self.policy_model.vec_obs_size > 0:
            self.obs_in_expert = tf.placeholder(
                shape=[None, self.policy_model.vec_obs_size], dtype=tf.float32)
            # TODO : Experiment with normalization, the normalization could change with time
            if self.policy_model.normalize:
                encoded_expert_list.append(self.policy_model.normalize_vector_obs(self.obs_in_expert))
                encoded_policy_list.append(self.policy_model.normalize_vector_obs(self.policy_model.vector_in))
            else:
                encoded_expert_list.append(self.obs_in_expert)
                encoded_policy_list.append(self.policy_model.vector_in)

        if self.policy_model.vis_obs_size > 0:
            self.expert_visual_in = []
            visual_policy_encoders = []
            visual_expert_encoders = []
            for i in range(self.policy_model.vis_obs_size):
                # Create input ops for next (t+1) visual observations.
                visual_input = self.policy_model.create_visual_input(
                    self.policy_model.brain.camera_resolutions[i],
                    name="visual_observation_" + str(i))
                self.expert_visual_in.append(visual_input)

                encoded_policy_visual = self.policy_model.create_visual_obs_encoder(
                    self.policy_model.visual_in[i],
                    self.encoding_size,
                    LearningModel.swish, 1,
                    "stream_{}_visual_obs_encoder"
                        .format(i), False)

                encoded_expert_visual = self.policy_model.create_visual_obs_encoder(
                    self.expert_visual_in[i],
                    self.encoding_size,
                    LearningModel.swish, 1,
                    "stream_{}_visual_obs_encoder".format(i),
                    True)
                visual_policy_encoders.append(encoded_policy_visual)
                visual_expert_encoders.append(encoded_expert_visual)
            hidden_policy_visual = tf.concat(visual_policy_encoders, axis=1)
            hidden_expert_visual = tf.concat(visual_expert_encoders, axis=1)
            encoded_policy_list.append(hidden_policy_visual)
            encoded_expert_list.append(hidden_expert_visual)

        self.encoded_expert = tf.concat(encoded_expert_list, axis=1)
        self.encoded_policy = tf.concat(encoded_policy_list, axis=1)

    def create_encoder(self, state_in, action_in, done_in, reuse):
        """
        Creates the encoder for the discriminator
        :param state_in: The encoded observation input
        :param action_in: The action input
        :param done_in: The done flags input
        :param reuse: If true, the weights will be shared with the previous encoder created
        """
        with tf.variable_scope("model"):
            if self.use_actions:
                concat_input = tf.concat([state_in, action_in, done_in], axis=1)
            else:
                concat_input = state_in

            hidden_1 = tf.layers.dense(
                concat_input, self.h_size, activation=tf.nn.elu,
                name="d_hidden_1", reuse=reuse)

            hidden_2 = tf.layers.dense(
                hidden_1, self.h_size, activation=tf.nn.elu,
                name="d_hidden_2", reuse=reuse)

            z_mean = None
            if self.use_vail:
                # Latent representation
                z_mean = tf.layers.dense(
                    hidden_2,
                    self.z_size,
                    reuse=reuse,
                    name="z_mean",
                    kernel_initializer=LearningModel.scaled_init(0.01))

                self.noise = tf.random_normal(tf.shape(z_mean), dtype=tf.float32)

                # Sampled latent code
                self.z = z_mean + self.z_sigma * self.noise * self.use_noise
                estimate_input = self.z
            else:
                estimate_input = hidden_2

            estimate = tf.layers.dense(estimate_input, 1, activation=tf.nn.sigmoid,
                                       name="d_estimate", reuse=reuse)
            return estimate, z_mean

    def create_loss(self, learning_rate, max_step, decay_duration):
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
            self.loss = tf.reduce_sum(tf.squared_difference(selected_action, self.expert_action)) / self.n_sequences \
                        - tf.reduce_sum(entropy)
        else:
            log_probs = self.policy_model.all_log_probs
            # self.expert_action = tf.placeholder(shape=[None, len(action_size)], dtype=tf.int32)
            action_idx = [0] + list(np.cumsum(action_size))
            entropy = tf.reduce_sum((tf.stack([
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.nn.softmax(log_probs[:, action_idx[i]:action_idx[i + 1]]),
                    logits=log_probs[:, action_idx[i]:action_idx[i + 1]])
                for i in range(len(action_size))], axis=1)), axis=1)
            self.loss = tf.reduce_sum(-tf.log(tf.nn.softmax(log_probs)+ 0*1e-10) * self.expert_action) #- 1 * tf.reduce_sum(entropy)

        #self.loss = tf.train.polynomial_decay(self.loss, self.policy_model.global_step, max_step*decay_duration, 0.0, power=1.0)
        annealed_learning_rate = tf.train.polynomial_decay(learning_rate, self.policy_model.global_step, max_step*decay_duration, 0.0, power=1.0)

        optimizer = tf.train.AdamOptimizer(learning_rate=annealed_learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

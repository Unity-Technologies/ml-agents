import tensorflow as tf
from mlagents.trainers.models import LearningModel


class GAILModel(object):
    def __init__(self, policy_model, h_size, lr, encoding_size):
        self.h_size = h_size
        self.z_size = 32
        self.beta = 0.1
        self.policy_model = policy_model
        self.encoding_size = encoding_size
        self.use_vail = True
        self.use_actions = True
        self.make_inputs()
        self.create_network()
        self.create_loss(5e-5)

    def make_inputs(self):
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

            if self.use_vail:
                # Latent representation
                self.z_mean = tf.layers.dense(hidden_2, self.z_size, reuse=reuse)
                self.z_log_sigma_sq = tf.layers.dense(hidden_2, self.z_size, reuse=reuse)
                self.z_sigma_sq = tf.exp(self.z_log_sigma_sq)
                self.z_sigma = tf.sqrt(self.z_sigma_sq)
                self.noise = tf.random_normal(shape=[self.z_size])

                # Sampled latent code
                self.z = self.z_mean + self.z_sigma * self.noise
                estimate_input = self.z
            else:
                estimate_input = hidden_2

            estimate = tf.layers.dense(estimate_input, 1, activation=tf.nn.sigmoid,
                                       name="d_estimate", reuse=reuse)
            return estimate

    def create_network(self):
        self.expert_estimate = self.create_encoder(
            self.encoded_expert, self.expert_action, self.done_expert, False)
        self.policy_estimate = self.create_encoder(
            self.encoded_policy, self.policy_model.selected_actions, self.done_policy, True)
        self.discriminator_score = tf.reshape(self.policy_estimate, [-1], name="GAIL_reward")
        self.intrinsic_reward = -tf.log(1.0 - self.discriminator_score + 1e-7)

    def update_beta(self, kl_div):
        self.beta = max(0, self.beta + 1e-5 * (kl_div - 0.5))
        # print(self.beta, kl_div)

    def create_loss(self, learning_rate):
        self.mean_expert_estimate = tf.reduce_mean(self.expert_estimate)
        self.mean_policy_estimate = tf.reduce_mean(self.policy_estimate)

        self.disc_loss = -tf.reduce_mean(
            tf.log(self.expert_estimate + 1e-10) + tf.log(1.0 - self.policy_estimate + 1e-10))

        if self.use_vail:
            # KL divergence loss (encourage latent representation to be normal)
            self.kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(
                1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1))
            self.loss = self.beta * (self.kl_loss - 0.5) + self.disc_loss
        else:
            self.loss = self.disc_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

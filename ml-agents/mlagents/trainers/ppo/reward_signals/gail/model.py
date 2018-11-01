import tensorflow as tf
from mlagents.trainers.models import LearningModel


class GAILModel(object):
    def __init__(self, policy_model, h_size, lr, encoding_size):
        self.h_size = h_size
        self.policy_model = policy_model
        self.encoding_size = encoding_size
        self.make_inputs()
        self.create_network()
        self.create_loss(lr)

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
            concat_input = tf.concat([state_in, action_in, done_in], axis=1)

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
        self.expert_estimate = self.create_encoder(
            self.encoded_expert, self.expert_action, self.done_expert, False)
        self.policy_estimate = self.create_encoder(
            self.encoded_policy, self.policy_model.selected_actions, self.done_policy, True)
        self.discriminator_score = tf.reshape(self.policy_estimate, [-1], name="GAIL_reward")
        self.intrinsic_reward = -tf.log(1.0 - self.discriminator_score + 1e-7)

    def create_loss(self, learning_rate):
        self.mean_expert_estimate = tf.reduce_mean(self.expert_estimate)
        self.mean_policy_estimate = tf.reduce_mean(self.policy_estimate)
        self.loss = -tf.reduce_mean(
            tf.log(self.expert_estimate + 1e-10) + tf.log(1.0 - self.policy_estimate + 1e-10))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.loss)

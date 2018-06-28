import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

logger = logging.getLogger("unityagents")


class LearningModel(object):
    def __init__(self, m_size, normalize, use_recurrent, brain):
        self.brain = brain
        self.vector_in = None
        self.normalize = False
        self.use_recurrent = False
        self.global_step, self.increment_step = self.create_global_steps()
        self.visual_in = []
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(shape=None, dtype=tf.int32, name='sequence_length')
        self.mask_input = tf.placeholder(shape=[None], dtype=tf.float32, name='masks')
        self.mask = tf.cast(self.mask_input, tf.int32)
        self.m_size = m_size
        self.normalize = normalize
        self.use_recurrent = use_recurrent
        self.a_size = brain.vector_action_space_size
        self.o_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
        self.v_size = brain.number_visual_observations

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        increment_step = tf.assign(global_step, tf.add(global_step, 1))
        return global_step, increment_step

    @staticmethod
    def swish(input_activation):
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))

    @staticmethod
    def create_visual_input(camera_parameters, name):
        """
        Creates image input op.
        :param camera_parameters: Parameters for visual observation from BrainInfo.
        :param name: Desired name of input op.
        :return: input op.
        """
        o_size_h = camera_parameters['height']
        o_size_w = camera_parameters['width']
        bw = camera_parameters['blackAndWhite']

        if bw:
            c_channels = 1
        else:
            c_channels = 3

        visual_in = tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32, name=name)
        return visual_in

    def create_vector_input(self, name='vector_observation'):
        """
        Creates ops for vector observation input.
        :param name: Name of the placeholder op.
        :param o_size: Size of stacked vector observation.
        :return:
        """
        if self.brain.vector_observation_space_type == "continuous":
            self.vector_in = tf.placeholder(shape=[None, self.o_size], dtype=tf.float32, name=name)
            if self.normalize:
                self.running_mean = tf.get_variable("running_mean", [self.o_size], trainable=False, dtype=tf.float32,
                                                    initializer=tf.zeros_initializer())
                self.running_variance = tf.get_variable("running_variance", [self.o_size], trainable=False,
                                                        dtype=tf.float32, initializer=tf.ones_initializer())
                self.update_mean, self.update_variance = self.create_normalizer_update(self.vector_in)

                self.normalized_state = tf.clip_by_value((self.vector_in - self.running_mean) / tf.sqrt(
                    self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)), -5, 5,
                                                         name="normalized_state")
                return self.normalized_state
            else:
                return self.vector_in
        else:
            self.vector_in = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='vector_observation')
            return self.vector_in

    def create_normalizer_update(self, vector_input):
        mean_current_observation = tf.reduce_mean(vector_input, axis=0)
        new_mean = self.running_mean + (mean_current_observation - self.running_mean) / \
                   tf.cast(self.global_step + 1, tf.float32)
        new_variance = self.running_variance + (mean_current_observation - new_mean) * \
                       (mean_current_observation - self.running_mean)
        update_mean = tf.assign(self.running_mean, new_mean)
        update_variance = tf.assign(self.running_variance, new_variance)
        return update_mean, update_variance

    @staticmethod
    def create_continuous_observation_encoder(observation_input, h_size, activation, num_layers, scope, reuse):
        """
        Builds a set of hidden state encoders.
        :param reuse: Whether to re-use the weights within the same scope.
        :param scope: Graph scope for the encoder ops.
        :param observation_input: Input vector.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            hidden = observation_input
            for i in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, activation=activation, reuse=reuse, name="hidden_{}".format(i),
                                         kernel_initializer=c_layers.variance_scaling_initializer(1.0))
        return hidden

    def create_visual_observation_encoder(self, image_input, h_size, activation, num_layers, scope, reuse):
        """
        Builds a set of visual (CNN) encoders.
        :param reuse: Whether to re-use the weights within the same scope.
        :param scope: The scope of the graph within which to create the ops.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            conv1 = tf.layers.conv2d(image_input, 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=tf.nn.elu, reuse=reuse, name="conv_1")
            conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                     activation=tf.nn.elu, reuse=reuse, name="conv_2")
            hidden = c_layers.flatten(conv2)

        with tf.variable_scope(scope+'/'+'flat_encoding'):
            hidden_flat = self.create_continuous_observation_encoder(hidden, h_size, activation,
                                                                     num_layers, scope, reuse)
        return hidden_flat

    @staticmethod
    def create_discrete_observation_encoder(observation_input, s_size, h_size, activation,
                                            num_layers, scope, reuse):
        """
        Builds a set of hidden state encoders from discrete state input.
        :param reuse: Whether to re-use the weights within the same scope.
        :param scope: The scope of the graph within which to create the ops.
        :param observation_input: Discrete observation.
        :param s_size: state input size (discrete).
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            vector_in = tf.reshape(observation_input, [-1])
            state_onehot = tf.one_hot(vector_in, s_size)
            hidden = state_onehot
            for i in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation,
                                         reuse=reuse, name="hidden_{}".format(i))
        return hidden

    def create_observation_streams(self, num_streams, h_size, num_layers):
        """
        Creates encoding stream for observations.
        :param num_streams: Number of streams to create.
        :param h_size: Size of hidden linear layers in stream.
        :param num_layers: Number of hidden linear layers in stream.
        :return: List of encoded streams.
        """
        brain = self.brain
        activation_fn = self.swish

        self.visual_in = []
        for i in range(brain.number_visual_observations):
            visual_input = self.create_visual_input(brain.camera_resolutions[i], name="visual_observation_" + str(i))
            self.visual_in.append(visual_input)
        vector_observation_input = self.create_vector_input()

        final_hiddens = []
        for i in range(num_streams):
            visual_encoders = []
            hidden_state, hidden_visual = None, None
            if self.v_size > 0:
                for j in range(brain.number_visual_observations):
                    encoded_visual = self.create_visual_observation_encoder(self.visual_in[j], h_size,
                                                                            activation_fn, num_layers,
                                                                            "main_graph_{}_encoder{}"
                                                                            .format(i, j), False)
                    visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if brain.vector_observation_space_size > 0:
                if brain.vector_observation_space_type == "continuous":
                    hidden_state = self.create_continuous_observation_encoder(vector_observation_input,
                                                                              h_size, activation_fn, num_layers,
                                                                              "main_graph_{}".format(i), False)
                else:
                    hidden_state = self.create_discrete_observation_encoder(vector_observation_input, self.o_size,
                                                                            h_size, activation_fn, num_layers,
                                                                            "main_graph_{}".format(i), False)
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = tf.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state is not None and hidden_visual is None:
                final_hidden = hidden_state
            else:
                raise Exception("No valid network configuration possible. "
                                "There are no states or observations in this brain")
            final_hiddens.append(final_hidden)
        return final_hiddens

    @staticmethod
    def create_recurrent_encoder(input_state, memory_in, sequence_length, name='lstm'):
        """
        Builds a recurrent encoder for either state or observations (LSTM).
        :param sequence_length: Length of sequence to unroll.
        :param input_state: The input tensor to the LSTM cell.
        :param memory_in: The input memory to the LSTM cell.
        :param name: The scope of the LSTM cell.
        """
        s_size = input_state.get_shape().as_list()[1]
        m_size = memory_in.get_shape().as_list()[1]
        lstm_input_state = tf.reshape(input_state, shape=[-1, sequence_length, s_size])
        memory_in = tf.reshape(memory_in[:, :], [-1, m_size])
        _half_point = int(m_size / 2)
        with tf.variable_scope(name):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(_half_point)
            lstm_vector_in = tf.contrib.rnn.LSTMStateTuple(memory_in[:, :_half_point], memory_in[:, _half_point:])
            recurrent_output, lstm_state_out = tf.nn.dynamic_rnn(rnn_cell, lstm_input_state,
                                                                 initial_state=lstm_vector_in)

        recurrent_output = tf.reshape(recurrent_output, shape=[-1, _half_point])
        return recurrent_output, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

    def create_dc_actor_critic(self, h_size, num_layers):
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_streams = self.create_observation_streams(1, h_size, num_layers)
        hidden = hidden_streams[0]

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
            prev_action_oh = tf.one_hot(self.prev_action, self.a_size)
            hidden = tf.concat([hidden, prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden, memory_out = self.create_recurrent_encoder(hidden, self.memory_in, self.sequence_length)
            self.memory_out = tf.identity(memory_out, name='recurrent_out')

        self.policy = tf.layers.dense(hidden, self.a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.all_probs = tf.nn.softmax(self.policy, name="action_probs")
        output = tf.multinomial(self.policy, 1)
        self.output = tf.identity(output, name="action")

        value = tf.layers.dense(hidden, 1, activation=None)
        self.value = tf.identity(value, name="value_estimate")
        self.entropy = -tf.reduce_sum(self.all_probs * tf.log(self.all_probs + 1e-10), axis=1)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = tf.one_hot(self.action_holder, self.a_size)

        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name='old_probabilities')

        # We reshape these tensors to [batch x 1] in order to be of the same rank as continuous control probabilities.
        self.probs = tf.expand_dims(tf.reduce_sum(self.all_probs * self.selected_actions, axis=1), 1)
        self.old_probs = tf.expand_dims(tf.reduce_sum(self.all_old_probs * self.selected_actions, axis=1), 1)

    def create_cc_actor_critic(self, h_size, num_layers):
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_streams = self.create_observation_streams(2, h_size, num_layers)

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            _half_point = int(self.m_size / 2)
            hidden_policy, memory_policy_out = self.create_recurrent_encoder(
                hidden_streams[0], self.memory_in[:, :_half_point], self.sequence_length, name='lstm_policy')

            hidden_value, memory_value_out = self.create_recurrent_encoder(
                hidden_streams[1], self.memory_in[:, _half_point:], self.sequence_length, name='lstm_value')
            self.memory_out = tf.concat([memory_policy_out, memory_value_out], axis=1, name='recurrent_out')
        else:
            hidden_policy = hidden_streams[0]
            hidden_value = hidden_streams[1]

        mu = tf.layers.dense(hidden_policy, self.a_size, activation=None,
                             kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        log_sigma_sq = tf.get_variable("log_sigma_squared", [self.a_size], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())

        sigma_sq = tf.exp(log_sigma_sq)

        epsilon = tf.random_normal(tf.shape(mu), dtype=tf.float32)

        # Clip and scale output to ensure actions are always within [-1, 1] range.
        self.output_pre = mu + tf.sqrt(sigma_sq) * epsilon
        output_post = tf.clip_by_value(self.output_pre, -3, 3) / 3
        self.output = tf.identity(output_post, name='action')
        self.selected_actions = tf.stop_gradient(output_post)

        # Compute probability of model output.
        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output_pre) - mu, 2) / (2 * sigma_sq))
        b = 1 / tf.sqrt(2 * sigma_sq * np.pi)
        all_probs = tf.multiply(a, b)
        self.all_probs = tf.identity(all_probs, name='action_probs')

        self.entropy = tf.reduce_mean(0.5 * tf.log(2 * np.pi * np.e * sigma_sq))

        value = tf.layers.dense(hidden_value, 1, activation=None)
        self.value = tf.identity(value, name="value_estimate")

        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32,
                                            name='old_probabilities')

        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.probs = tf.identity(self.all_probs)
        self.old_probs = tf.identity(self.all_old_probs)

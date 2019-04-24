import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

logger = logging.getLogger("mlagents.trainers")


class LearningModel(object):
    _version_number_ = 2

    def __init__(self, m_size, normalize, use_recurrent, brain, seed):
        tf.set_random_seed(seed)
        self.brain = brain
        self.vector_in = None
        self.global_step, self.increment_step = self.create_global_steps()
        self.visual_in = []
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name="batch_size")
        self.sequence_length = tf.placeholder(
            shape=None, dtype=tf.int32, name="sequence_length"
        )
        self.mask_input = tf.placeholder(shape=[None], dtype=tf.float32, name="masks")
        self.mask = tf.cast(self.mask_input, tf.int32)
        self.use_recurrent = use_recurrent
        if self.use_recurrent:
            self.m_size = m_size
        else:
            self.m_size = 0
        self.normalize = normalize
        self.act_size = brain.vector_action_space_size
        self.vec_obs_size = (
            brain.vector_observation_space_size * brain.num_stacked_vector_observations
        )
        self.vis_obs_size = brain.number_visual_observations
        tf.Variable(
            int(brain.vector_action_space_type == "continuous"),
            name="is_continuous_control",
            trainable=False,
            dtype=tf.int32,
        )
        tf.Variable(
            self._version_number_,
            name="version_number",
            trainable=False,
            dtype=tf.int32,
        )
        tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
        if brain.vector_action_space_type == "continuous":
            tf.Variable(
                self.act_size[0],
                name="action_output_shape",
                trainable=False,
                dtype=tf.int32,
            )
        else:
            tf.Variable(
                sum(self.act_size),
                name="action_output_shape",
                trainable=False,
                dtype=tf.int32,
            )

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(
            0, name="global_step", trainable=False, dtype=tf.int32
        )
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
        o_size_h = camera_parameters["height"]
        o_size_w = camera_parameters["width"]
        bw = camera_parameters["blackAndWhite"]

        if bw:
            c_channels = 1
        else:
            c_channels = 3

        visual_in = tf.placeholder(
            shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32, name=name
        )
        return visual_in

    def create_vector_input(self, name="vector_observation"):
        """
        Creates ops for vector observation input.
        :param name: Name of the placeholder op.
        :param vec_obs_size: Size of stacked vector observation.
        :return:
        """
        self.vector_in = tf.placeholder(
            shape=[None, self.vec_obs_size], dtype=tf.float32, name=name
        )
        if self.normalize:
            self.running_mean = tf.get_variable(
                "running_mean",
                [self.vec_obs_size],
                trainable=False,
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
            )
            self.running_variance = tf.get_variable(
                "running_variance",
                [self.vec_obs_size],
                trainable=False,
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
            )
            self.update_mean, self.update_variance = self.create_normalizer_update(
                self.vector_in
            )

            self.normalized_state = tf.clip_by_value(
                (self.vector_in - self.running_mean)
                / tf.sqrt(
                    self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)
                ),
                -5,
                5,
                name="normalized_state",
            )
            return self.normalized_state
        else:
            return self.vector_in

    def create_normalizer_update(self, vector_input):
        mean_current_observation = tf.reduce_mean(vector_input, axis=0)
        new_mean = self.running_mean + (
            mean_current_observation - self.running_mean
        ) / tf.cast(tf.add(self.global_step, 1), tf.float32)
        new_variance = self.running_variance + (mean_current_observation - new_mean) * (
            mean_current_observation - self.running_mean
        )
        update_mean = tf.assign(self.running_mean, new_mean)
        update_variance = tf.assign(self.running_variance, new_variance)
        return update_mean, update_variance

    @staticmethod
    def create_vector_observation_encoder(
        observation_input, h_size, activation, num_layers, scope, reuse
    ):
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
                hidden = tf.layers.dense(
                    hidden,
                    h_size,
                    activation=activation,
                    reuse=reuse,
                    name="hidden_{}".format(i),
                    kernel_initializer=c_layers.variance_scaling_initializer(1.0),
                )
        return hidden

    def create_visual_observation_encoder(
        self, image_input, h_size, activation, num_layers, scope, reuse
    ):
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
            conv1 = tf.layers.conv2d(
                image_input,
                16,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_1",
            )
            conv2 = tf.layers.conv2d(
                conv1,
                32,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_2",
            )
            hidden = c_layers.flatten(conv2)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = self.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def create_discrete_action_masking_layer(all_logits, action_masks, action_size):
        """
        Creates a masking layer for the discrete actions
        :param all_logits: The concatenated unnormalized action probabilities for all branches
        :param action_masks: The mask for the logits. Must be of dimension [None x total_number_of_action]
        :param action_size: A list containing the number of possible actions for each branch
        :return: The action output dimension [batch_size, num_branches] and the concatenated normalized logits
        """
        action_idx = [0] + list(np.cumsum(action_size))
        branches_logits = [
            all_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        branch_masks = [
            action_masks[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        raw_probs = [
            tf.multiply(tf.nn.softmax(branches_logits[k]) + 1.0e-10, branch_masks[k])
            for k in range(len(action_size))
        ]
        normalized_probs = [
            tf.divide(raw_probs[k], tf.reduce_sum(raw_probs[k], axis=1, keepdims=True))
            for k in range(len(action_size))
        ]
        output = tf.concat(
            [
                tf.multinomial(tf.log(normalized_probs[k]), 1)
                for k in range(len(action_size))
            ],
            axis=1,
        )
        return (
            output,
            tf.concat(
                [
                    tf.log(normalized_probs[k] + 1.0e-10)
                    for k in range(len(action_size))
                ],
                axis=1,
            ),
        )

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
            visual_input = self.create_visual_input(
                brain.camera_resolutions[i], name="visual_observation_" + str(i)
            )
            self.visual_in.append(visual_input)
        vector_observation_input = self.create_vector_input()

        final_hiddens = []
        for i in range(num_streams):
            visual_encoders = []
            hidden_state, hidden_visual = None, None
            if self.vis_obs_size > 0:
                for j in range(brain.number_visual_observations):
                    encoded_visual = self.create_visual_observation_encoder(
                        self.visual_in[j],
                        h_size,
                        activation_fn,
                        num_layers,
                        "main_graph_{}_encoder{}".format(i, j),
                        False,
                    )
                    visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if brain.vector_observation_space_size > 0:
                hidden_state = self.create_vector_observation_encoder(
                    vector_observation_input,
                    h_size,
                    activation_fn,
                    num_layers,
                    "main_graph_{}".format(i),
                    False,
                )
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = tf.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state is not None and hidden_visual is None:
                final_hidden = hidden_state
            else:
                raise Exception(
                    "No valid network configuration possible. "
                    "There are no states or observations in this brain"
                )
            final_hiddens.append(final_hidden)
        return final_hiddens

    @staticmethod
    def create_recurrent_encoder(input_state, memory_in, sequence_length, name="lstm"):
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
            lstm_vector_in = tf.contrib.rnn.LSTMStateTuple(
                memory_in[:, :_half_point], memory_in[:, _half_point:]
            )
            recurrent_output, lstm_state_out = tf.nn.dynamic_rnn(
                rnn_cell, lstm_input_state, initial_state=lstm_vector_in
            )

        recurrent_output = tf.reshape(recurrent_output, shape=[-1, _half_point])
        return recurrent_output, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

    def create_cc_actor_critic(self, h_size, num_layers):
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_streams = self.create_observation_streams(2, h_size, num_layers)

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
            kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01),
        )

        log_sigma_sq = tf.get_variable(
            "log_sigma_squared",
            [self.act_size[0]],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
        )

        sigma_sq = tf.exp(log_sigma_sq)

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
            - 0.5 * log_sigma_sq
        )

        self.all_log_probs = tf.identity(all_probs, name="action_probs")

        self.entropy = 0.5 * tf.reduce_mean(tf.log(2 * np.pi * np.e) + log_sigma_sq)

        value = tf.layers.dense(hidden_value, 1, activation=None)
        self.value = tf.identity(value, name="value_estimate")

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

    def create_dc_actor_critic(self, h_size, num_layers):
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        hidden_streams = self.create_observation_streams(1, h_size, num_layers)
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
                    kernel_initializer=c_layers.variance_scaling_initializer(
                        factor=0.01
                    ),
                )
            )

        self.all_log_probs = tf.concat(
            [branch for branch in policy_branches], axis=1, name="action_probs"
        )

        self.action_masks = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
        )
        output, normalized_logits = self.create_discrete_action_masking_layer(
            self.all_log_probs, self.action_masks, self.act_size
        )

        self.output = tf.identity(output)
        self.normalized_logits = tf.identity(normalized_logits, name="action")

        value = tf.layers.dense(hidden, 1, activation=None)
        self.value = tf.identity(value, name="value_estimate")

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
        _, old_normalized_logits = self.create_discrete_action_masking_layer(
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

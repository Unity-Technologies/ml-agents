import logging

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

from tensorflow.python.tools import freeze_graph
from unityagents import UnityEnvironmentException

logger = logging.getLogger("unityagents")


def create_agent_model(brain, lr=1e-4, h_size=128, epsilon=0.2, beta=1e-3, max_step=5e6,
                       normalize=False, use_recurrent=False, num_layers=2, m_size=None):
    """
    Takes a Unity environment and model-specific hyper-parameters and returns the
    appropriate PPO agent model for the environment.
    :param brain: BrainInfo used to generate specific network graph.
    :param lr: Learning rate.
    :param h_size: Size of hidden layers/
    :param epsilon: Value for policy-divergence threshold.
    :param beta: Strength of entropy regularization.
    :return: a sub-class of PPOAgent tailored to the environment.
    :param max_step: Total number of training steps.
    :param normalize: Whether to normalize vector observation input.
    :param use_recurrent: Whether to use an LSTM layer in the network.
    :param num_layers Number of hidden layers between encoded input and policy & value layers
    """

    if num_layers < 1:
        num_layers = 1

    if brain.action_space_type == "continuous":
        return ContinuousControlModel(lr, brain, h_size, epsilon, max_step, normalize, use_recurrent, num_layers,
                                      m_size)
    if brain.action_space_type == "discrete":
        return DiscreteControlModel(lr, brain, h_size, epsilon, beta, max_step, normalize, use_recurrent, num_layers,
                                    m_size)


def save_model(sess, saver, model_path="./", steps=0):
    """
    Saves current model to checkpoint folder.
    :param sess: Current Tensorflow session.
    :param model_path: Designated model path.
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    last_checkpoint = model_path + '/model-' + str(steps) + '.cptk'
    saver.save(sess, last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    logger.info("Saved Model")


def export_graph(model_path, env_name="env", target_nodes="action,value_estimate,action_probs"):
    """
    Exports latest saved model to .bytes format for Unity embedding.
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    """
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")


class PPOModel(object):
    def __init__(self, m_size, normalize, use_recurrent):
        self.normalize = False
        self.use_recurrent = False
        self.observation_in = []
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(shape=None, dtype=tf.int32, name='sequence_length')
        self.m_size = m_size
        self.global_step, self.increment_step = self.create_global_steps()
        self.last_reward, self.new_reward, self.update_reward = self.create_reward_encoder()
        self.normalize = normalize
        self.use_recurrent = use_recurrent
        self.state_in = None

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        increment_step = tf.assign(global_step, tf.add(global_step, 1))
        return global_step, increment_step

    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    def create_recurrent_encoder(self, input_state, memory_in, name = 'lstm'):
        """
        Builds a recurrent encoder for either state or observations (LSTM).
        :param input_state: The input tensor to the LSTM cell.
        :param memory_in: The input memory to the LSTM cell.
        :param name: The scope of the LSTM cell.
        """
        s_size = input_state.get_shape().as_list()[1]
        m_size = memory_in.get_shape().as_list()[1]
        lstm_input_state = tf.reshape(input_state, shape = [-1, self.sequence_length, s_size])
        _half_point = int(m_size/2)
        with tf.variable_scope(name):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(_half_point)
            lstm_state_in = tf.contrib.rnn.LSTMStateTuple(memory_in[:,:_half_point], memory_in[:,_half_point:])
            recurrent_state, lstm_state_out = tf.nn.dynamic_rnn(rnn_cell, lstm_input_state,
                                       initial_state=lstm_state_in,
                                        time_major=False,
                                       dtype=tf.float32)
        recurrent_state = tf.reshape(recurrent_state, shape = [-1, _half_point])
        return recurrent_state, tf.concat([lstm_state_out.c, lstm_state_out.h], axis = 1)

    def create_visual_encoder(self, o_size_h, o_size_w, bw, h_size, num_streams, activation, num_layers):
        """
        Builds a set of visual (CNN) encoders.
        :param o_size_h: Height observation size.
        :param o_size_w: Width observation size.
        :param bw: Whether image is greyscale {True} or color {False}.
        :param h_size: Hidden layer size.
        :param num_streams: Number of visual streams to construct.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        if bw:
            c_channels = 1
        else:
            c_channels = 3

        self.observation_in.append(tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32,
                                                  name='observation_%d' % len(self.observation_in)))

        streams = []
        for i in range(num_streams):
            conv1 = tf.layers.conv2d(self.observation_in[-1], 16, kernel_size=[8, 8], strides=[4, 4],
                                     activation=activation)
            conv2 = tf.layers.conv2d(conv1, 32, kernel_size=[4, 4], strides=[2, 2],
                                     activation=activation)
            hidden = c_layers.flatten(conv2)

            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
                streams.append(hidden)
        return streams

    def create_continuous_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders.
        :param s_size: state input size.
        :param h_size: Hidden layer size.
        :param num_streams: Number of state streams to construct.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='state')

        if self.normalize:
            self.running_mean = tf.get_variable("running_mean", [s_size], trainable=False, dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
            self.running_variance = tf.get_variable("running_variance", [s_size], trainable=False, dtype=tf.float32,
                                                    initializer=tf.ones_initializer())

            self.normalized_state = tf.clip_by_value((self.state_in - self.running_mean) / tf.sqrt(
                self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)), -5, 5, name="normalized_state")

            self.new_mean = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_mean')
            self.new_variance = tf.placeholder(shape=[s_size], dtype=tf.float32, name='new_variance')
            self.update_mean = tf.assign(self.running_mean, self.new_mean)
            self.update_variance = tf.assign(self.running_variance, self.new_variance)
        else:
            self.normalized_state = self.state_in

        streams = []
        for i in range(num_streams):
            hidden = self.normalized_state
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, activation=activation,
                                         kernel_initializer=c_layers.variance_scaling_initializer(1.0))
            streams.append(hidden)
        return streams

    def create_discrete_state_encoder(self, s_size, h_size, num_streams, activation, num_layers):
        """
        Builds a set of hidden state encoders from discrete state input.
        :param s_size: state input size (discrete).
        :param h_size: Hidden layer size.
        :param num_streams: Number of state streams to construct.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        self.state_in = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='state')
        state_in = tf.reshape(self.state_in, [-1])
        state_onehot = c_layers.one_hot_encoding(state_in, s_size)
        streams = []

        hidden = state_onehot
        for i in range(num_streams):
            for j in range(num_layers):
                hidden = tf.layers.dense(hidden, h_size, use_bias=False, activation=activation)
            streams.append(hidden)
        return streams

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
        self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantages')

        decay_epsilon = tf.train.polynomial_decay(epsilon, self.global_step,
                                                  max_step, 0.1,
                                                  power=1.0)

        r_theta = probs / (old_probs + 1e-10)
        p_opt_a = r_theta * self.advantage
        p_opt_b = tf.clip_by_value(r_theta, 1 - decay_epsilon, 1 + decay_epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

        self.value_loss = tf.reduce_mean(tf.squared_difference(self.returns_holder,
                                                               tf.reduce_sum(value, axis=1)))

        decay_beta = tf.train.polynomial_decay(beta, self.global_step,
                                               max_step, 1e-5,
                                               power=1.0)

        self.loss = self.policy_loss + 0.5 * self.value_loss - decay_beta * tf.reduce_mean(entropy)

        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step,
                                                       max_step, 1e-10,
                                                       power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_batch = optimizer.minimize(self.loss)


class ContinuousControlModel(PPOModel):
    def __init__(self, lr, brain, h_size, epsilon, max_step, normalize, use_recurrent, num_layers, m_size):
        """
        Creates Continuous Control Actor-Critic model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        """
        super(ContinuousControlModel, self).__init__(m_size, normalize, use_recurrent)
        a_size = brain.action_space_size

        hidden_state, hidden_visual, hidden_policy, hidden_value = None, None, None, None
        if brain.number_observations > 0:
            visual_encoder_0 = []
            visual_encoder_1 = []
            for i in range(brain.number_observations):
                height_size, width_size = brain.camera_resolutions[i]['height'], brain.camera_resolutions[i]['width']
                bw = brain.camera_resolutions[i]['blackAndWhite']
                encoded_visual = self.create_visual_encoder(height_size, width_size, bw, h_size, 2, tf.nn.tanh,
                                                            num_layers)
                visual_encoder_0.append(encoded_visual[0])
                visual_encoder_1.append(encoded_visual[1])
            hidden_visual = [tf.concat(visual_encoder_0, axis=1), tf.concat(visual_encoder_1, axis=1)]
        if brain.state_space_size > 0:
            s_size = brain.state_space_size * brain.stacked_states
            if brain.state_space_type == "continuous":
                hidden_state = self.create_continuous_state_encoder(s_size, h_size, 2, tf.nn.tanh, num_layers)
            else:
                hidden_state = self.create_discrete_state_encoder(s_size, h_size, 2, tf.nn.tanh, num_layers)

        if hidden_visual is None and hidden_state is None:
            raise Exception("No valid network configuration possible. "
                            "There are no states or observations in this brain")
        elif hidden_visual is not None and hidden_state is None:
            hidden_policy, hidden_value = hidden_visual
        elif hidden_visual is None and hidden_state is not None:
            hidden_policy, hidden_value = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden_policy = tf.concat([hidden_visual[0], hidden_state[0]], axis=1)
            hidden_value = tf.concat([hidden_visual[1], hidden_state[1]], axis=1)

        if self.use_recurrent:
            self.memory_in = tf.placeholder(shape=[None, self.m_size],dtype=tf.float32, name='recurrent_in')
            _half_point = int(self.m_size/2)
            hidden_policy , memory_policy_out = self.create_recurrent_encoder( hidden_policy, self.memory_in[:, :_half_point ], name = 'lstm_policy')
            hidden_value , memory_value_out = self.create_recurrent_encoder( hidden_value, self.memory_in[:, _half_point: ], name = 'lstm_value')
            self.memory_out = tf.concat([memory_policy_out, memory_value_out], axis=1, name = 'recurrent_out')
        
        self.mu = tf.layers.dense(hidden_policy, a_size, activation=None, use_bias=False,

                                  kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.log_sigma_sq = tf.get_variable("log_sigma_squared", [a_size], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
        self.sigma_sq = tf.exp(self.log_sigma_sq)

        self.epsilon = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='epsilon')

        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output = tf.identity(self.output, name='action')

        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.probs = tf.multiply(a, b, name="action_probs")

        self.entropy = tf.reduce_sum(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))

        self.value = tf.layers.dense(hidden_value, 1, activation=None)

        self.value = tf.identity(self.value, name="value_estimate")

        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')

        self.create_ppo_optimizer(self.probs, self.old_probs, self.value, self.entropy, 0.0, epsilon, lr, max_step)


class DiscreteControlModel(PPOModel):
    def __init__(self, lr, brain, h_size, epsilon, beta, max_step, normalize, use_recurrent, num_layers, m_size):
        """
        Creates Discrete Control Actor-Critic model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        """
        super(DiscreteControlModel, self).__init__(m_size, normalize, use_recurrent)
        a_size = brain.action_space_size

        hidden_state, hidden_visual, hidden = None, None, None
        if brain.number_observations > 0:
            visual_encoders = []
            for i in range(brain.number_observations):
                height_size, width_size = brain.camera_resolutions[i]['height'], brain.camera_resolutions[i]['width']
                bw = brain.camera_resolutions[i]['blackAndWhite']
                visual_encoders.append(
                    self.create_visual_encoder(height_size, width_size, bw, h_size, 2, tf.nn.tanh, num_layers)[0])
            hidden_visual = [tf.concat(visual_encoders, axis=1)]
        if brain.state_space_size > 0:
            s_size = brain.state_space_size * brain.stacked_states
            if brain.state_space_type == "continuous":
                hidden_state = \
                    self.create_continuous_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]
            else:
                hidden_state = self.create_discrete_state_encoder(s_size, h_size, 1, tf.nn.elu, num_layers)[0]

        if hidden_visual is None and hidden_state is None:
            raise Exception("No valid network configuration possible. "
                            "There are no states or observations in this brain")
        elif hidden_visual is not None and hidden_state is None:
            hidden = hidden_visual[0]
        elif hidden_visual is None and hidden_state is not None:
            hidden = hidden_state
        elif hidden_visual is not None and hidden_state is not None:
            hidden = tf.concat([hidden_visual[0], hidden_state], axis=1)

        if self.use_recurrent:
            self.memory_in = tf.placeholder(shape=[None, self.m_size],dtype=tf.float32, name='recurrent_in')
            hidden, self.memory_out = self.create_recurrent_encoder( hidden, self.memory_in)
            self.memory_out = tf.identity(self.memory_out,  name = 'recurrent_out')

        a_size = brain.action_space_size

        self.policy = tf.layers.dense(hidden, a_size, activation=None, use_bias=False,

                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))
        self.probs = tf.nn.softmax(self.policy, name="action_probs")
        self.output = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.output, name="action")
        self.value = tf.layers.dense(hidden, 1, activation=None)
        self.value = tf.identity(self.value, name="value_estimate")

        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs + 1e-10), axis=1)

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, a_size)
        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')
        self.responsible_probs = tf.reduce_sum(self.probs * self.selected_actions, axis=1)
        self.old_responsible_probs = tf.reduce_sum(self.old_probs * self.selected_actions, axis=1)

        self.create_ppo_optimizer(self.responsible_probs, self.old_responsible_probs,
                                  self.value, self.entropy, beta, epsilon, lr, max_step)

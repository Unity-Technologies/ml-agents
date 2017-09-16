import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers
from tensorflow.python.tools import freeze_graph


def create_agent_model(env, lr=1e-4, h_size=128, epsilon=0.2, beta=1e-3):
    """
    Takes a Unity environment and model-specific hyperparameters and returns the
    appropriate PPO agent model for the environment.
    :param env: a Unity environment.
    :param lr: Learning rate.
    :param h_size: Size of hidden layers/
    :param epsilon: Value for policy-divergence threshold.
    :param beta: Strength of entropy regularization.
    :return: a sub-class of PPOAgent tailored to the environment.
    """
    brain_name = env.brain_names[0]
    if env.brains[brain_name].action_space_type == "continuous":
        return ContinuousControlModel(lr, env.brains[brain_name].state_space_size,
                                      env.brains[brain_name].action_space_size, h_size, epsilon, beta)
    if env.brains[brain_name].action_space_type == "discrete":
        if env.brains[brain_name].number_observations == 0:
            return DiscreteControlModel(lr, env.brains[brain_name].state_space_size,
                                        env.brains[brain_name].action_space_size, h_size, epsilon, beta)
        else:
            brain = env.brains[brain_name]
            h, w = brain.camera_resolutions[0]['height'], brain.camera_resolutions[0]['height']
            return VisualDiscreteControlModel(lr, h, w, env.brains[brain_name].action_space_size, h_size, epsilon, beta)


def save_model(sess, saver, model_path="./", steps=0):
    """
    Saves current model to checkpoint folder.
    :param sess: Current Tensorflow session.
    :param model_path: Designated model path.
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    last_checkpoint = model_path+'/model-'+str(steps)+'.cptk'
    saver.save(sess, last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    print("Saved Model")


def export_graph(model_path, env_name="env", target_nodes="action"):
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
    def __init__(self, probs, old_probs, value, entropy, beta, epsilon, lr):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value: Current value estimate
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        """
        self.returns_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='discounted_rewards')
        self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantages')

        r_theta = probs / old_probs
        p_opt_a = r_theta * self.advantage
        p_opt_b = tf.clip_by_value(r_theta, 1 - epsilon, 1 + epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))

        self.value_loss = tf.reduce_mean(tf.squared_difference(self.returns_holder,
                                                               tf.reduce_sum(value, axis=1)))

        self.loss = self.policy_loss + self.value_loss - beta * tf.reduce_mean(entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.minimize(self.loss)

        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        self.increment_step = tf.assign(self.global_step, self.global_step+1)


class ContinuousControlModel(PPOModel):
    def __init__(self, lr, s_size, a_size, h_size, epsilon, beta):
        """
        Creates Continuous Control Actor-Critic model.
        :param s_size: State-space size
        :param a_size: Action-space size
        :param h_size: Hidden layer size
        """
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='state')
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        hidden_policy = tf.layers.dense(self.state_in, h_size, use_bias=False, activation=tf.nn.tanh)
        hidden_value = tf.layers.dense(self.state_in, h_size, use_bias=False, activation=tf.nn.tanh)
        hidden_policy_2 = tf.layers.dense(hidden_policy, h_size, use_bias=False, activation=tf.nn.tanh)
        hidden_value_2 = tf.layers.dense(hidden_value, h_size, use_bias=False, activation=tf.nn.tanh)

        self.mu = tf.layers.dense(hidden_policy_2, a_size, activation=None, use_bias=False,
                                  kernel_initializer=c_layers.variance_scaling_initializer(factor=0.1))
        self.log_sigma_sq = tf.Variable(tf.zeros([a_size]))
        self.sigma_sq = tf.exp(self.log_sigma_sq)

        self.epsilon = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='epsilon')

        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output = tf.identity(self.output, name='action')

        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.probs = a * b

        self.entropy = tf.reduce_sum(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))

        self.value = tf.layers.dense(hidden_value_2, 1, activation=None, use_bias=False)

        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')

        PPOModel.__init__(self, self.probs, self.old_probs, self.value, self.entropy, 0.0, epsilon, lr)


class DiscreteControlModel(PPOModel):
    def __init__(self, lr, s_size, a_size, h_size, epsilon, beta):
        """
        Creates Discrete Control Actor-Critic model.
        :param s_size: State-space size
        :param a_size: Action-space size
        :param h_size: Hidden layer size
        """
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='state')
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        hidden_1 = tf.layers.dense(self.state_in, h_size, use_bias=False, activation=tf.nn.elu)
        hidden_2 = tf.layers.dense(hidden_1, h_size, use_bias=False, activation=tf.nn.elu)
        self.policy = tf.layers.dense(hidden_2, a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.1))
        self.probs = tf.nn.softmax(self.policy)
        self.action = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.action, name='action')
        self.value = tf.layers.dense(hidden_2, 1, activation=None, use_bias=False)

        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs + 1e-10), axis=1)

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, a_size)
        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')
        self.responsible_probs = tf.reduce_sum(self.probs * self.selected_actions, axis=1)
        self.old_responsible_probs = tf.reduce_sum(self.old_probs * self.selected_actions, axis=1)

        PPOModel.__init__(self, self.responsible_probs, self.old_responsible_probs,
                          self.value, self.entropy, beta, epsilon, lr)


class VisualDiscreteControlModel(PPOModel):
    def __init__(self, lr, o_size_h, o_size_w, a_size, h_size, epsilon, beta):
        """
        Creates Discrete Control Actor-Critic model for use with visual observations (images).
        :param o_size_h: Observation height.
        :param o_size_w: Observation width.
        :param a_size: Action-space size.
        :param h_size: Hidden layer size.
        """
        self.observation_in = tf.placeholder(shape=[None, o_size_h, o_size_w, 1], dtype=tf.float32,
                                             name='observation_0')
        self.conv1 = tf.layers.conv2d(self.observation_in, 32, kernel_size=[3, 3], strides=[2, 2],
                                      use_bias=False, activation=tf.nn.elu)
        self.conv2 = tf.layers.conv2d(self.conv1, 64, kernel_size=[3, 3], strides=[2, 2],
                                      use_bias=False, activation=tf.nn.elu)
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32)
        hidden = tf.layers.dense(c_layers.flatten(self.conv2), h_size, use_bias=False, activation=tf.nn.elu)
        self.policy = tf.layers.dense(hidden, a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.1))
        self.probs = tf.nn.softmax(self.policy)
        self.action = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.action, name='action')
        self.value = tf.layers.dense(hidden, 1, activation=None, use_bias=False)

        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs + 1e-10), axis=1)

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, a_size)
        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')
        self.responsible_probs = tf.reduce_sum(self.probs * self.selected_actions, axis=1)
        self.old_responsible_probs = tf.reduce_sum(self.old_probs * self.selected_actions, axis=1)

        PPOModel.__init__(self, self.responsible_probs, self.old_responsible_probs,
                          self.value, self.entropy, beta, epsilon, lr)

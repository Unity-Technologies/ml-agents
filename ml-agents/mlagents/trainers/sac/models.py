import logging
import numpy as np

import tensorflow as tf
from mlagents.trainers.models import LearningModel
import tensorflow.contrib.layers as c_layers

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6  # Small value to avoid divide by zero

logger = logging.getLogger("mlagents.trainers")


class SACNetwork(LearningModel):
    def __init__(
        self,
        brain,
        m_size=None,
        h_size=128,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        stream_names=None,
        seed=0,
        is_target=False,
    ):
        LearningModel.__init__(
            self, m_size, normalize, use_recurrent, brain, seed, stream_names
        )
        self.normalize = normalize
        self.use_recurrent = use_recurrent
        self.num_layers = num_layers
        self.stream_names = stream_names
        self.h_size = h_size
        if is_target:
            with tf.variable_scope("target_network"):
                hidden_streams = self.create_observation_streams(1, self.h_size, 1)
            self.create_cc_critic(hidden_streams[0], "target_network", create_qs=False)
        else:
            with tf.variable_scope("policy_network"):
                hidden_streams = self.create_observation_streams(2, self.h_size, 1)
            self.create_cc_actor(hidden_streams[0], "policy_network")
            self.create_cc_critic(hidden_streams[1], "policy_network")

    def get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def create_cc_critic(self, hidden_value, scope, create_qs = True):
        """
        Creates just the critic network
        """
        scope = scope + "/critic"
        self.value = self.create_sac_value_head(
            hidden_value, self.num_layers, self.h_size, scope + "/value"
        )
        
        self.value_vars = self.get_vars(scope + "/value")
        
        if create_qs:
            self.action_holder = tf.placeholder(
                shape=[None, self.act_size[0]], dtype=tf.float32, name="action_holder"
            )
            hidden_q = tf.concat([hidden_value, self.action_holder], axis = -1)
            hidden_qp = tf.concat([hidden_value, self.output_pre], axis = -1)
            self.q1_heads, self.q2_heads, self.q1, self.q2 = self.create_q_heads(
                self.stream_names, hidden_q, self.num_layers, self.h_size, scope + "/q"
            )
            self.q1_pheads, self.q2_pheads, self.q1_p, self.q2_p = self.create_q_heads(
                self.stream_names, hidden_qp, self.num_layers, self.h_size, scope + "/qp"
            )
            self.q_vars = self.get_vars(scope)
        self.critic_vars = self.get_vars(scope)

    def create_cc_actor(self, hidden_policy, scope):
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        """
        scope = scope + "/policy"
        hidden_policy = self.create_vector_observation_encoder(
            hidden_policy, self.h_size, self.swish, self.num_layers, scope, False
        )
        # hidden_policy = tf.Print(hidden_policy,[hidden_policy], name="HiddenPoicy" )
        with tf.variable_scope(scope):
            mu = tf.layers.dense(
                hidden_policy,
                self.act_size[0],
                activation=None
                # kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01),
            )

            # Policy-dependent log_sigma_sq
            log_sigma_sq = tf.layers.dense(
                hidden_policy,
                self.act_size[0],
                activation=None,
                # kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01),
            )

            self.log_sigma_sq = tf.clip_by_value(log_sigma_sq, LOG_STD_MIN, LOG_STD_MAX)

            # self.log_sigma_sq = tf.get_variable("log_sigma_squared", [self.act_size[0]], dtype=tf.float32,
            #                                initializer=tf.zeros_initializer())

            sigma_sq = tf.exp(self.log_sigma_sq)

            # Do the reparameterization trick
            policy_ = mu + tf.random_normal(tf.shape(mu)) * sigma_sq

            _gauss_pre = -0.5 * (
                ((policy_ - mu) / (tf.exp(self.log_sigma_sq) + EPSILON)) ** 2
                + 2 * self.log_sigma_sq
                + np.log(2 * np.pi)
            )
            all_probs = tf.reduce_sum(_gauss_pre, axis=1)

            # Squash probabilities
            # Keep deterministic around in case we want to use it.
            self.deterministic_output_pre = tf.tanh(mu)
            self.output_pre = tf.tanh(policy_)

            # Squash correction
            all_probs -= tf.reduce_sum(tf.log(1 - self.output_pre ** 2 + EPSILON), axis=1)

            # Clip and scale output to ensure actions are always within [-1, 1] range.
            output_post = tf.clip_by_value(self.output_pre, -3, 3) / 3
            self.output = tf.identity(output_post, name="action")
            self.selected_actions = tf.stop_gradient(output_post)

            # self.output = tf.Print(self.selected_actions, [all_probs, 1 - policy_ ** 2 + EPSILON], message = "log probs")


            self.all_log_probs = tf.identity(all_probs, name="action_probs")

            self.entropy = 0.5 * tf.reduce_mean(
                tf.log(2 * np.pi * np.e) + self.log_sigma_sq
            )

        # Get all policy vars
        self.policy_vars = self.get_vars(scope)

    def create_sac_value_head(self, hidden_input, num_layers, h_size, scope):
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        
        value_hidden = self.create_vector_observation_encoder(
            hidden_input, h_size, self.swish, num_layers, scope, False
        )
        with tf.variable_scope(scope):
            value = tf.layers.dense(
                    value_hidden,
                    1,
                    activation=self.swish,
                    reuse=False,
                    name="hidden_value",
                    kernel_initializer=c_layers.variance_scaling_initializer(1.0),
                )
        return value

    def create_q_heads(self, stream_names, hidden_input, num_layers, h_size, scope):
        """
        Creates two q heads for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        with tf.variable_scope(scope + "/" + "q1_encoding"):
            q1_hidden = self.create_vector_observation_encoder(
                hidden_input, h_size, self.swish, num_layers, "q1_encoder", False
            )
            q1_heads = {}
            for name in stream_names:
                _q1 = tf.layers.dense(q1_hidden, 1, name="{}_q1".format(name))
                q1_heads[name] = _q1
        with tf.variable_scope(scope + "/" + "q2_encoding"):
            q2_hidden = self.create_vector_observation_encoder(
                hidden_input, h_size, self.swish, num_layers, "q2_encoder", False
            )
            q2_heads = {}
            for name in stream_names:
                _q2 = tf.layers.dense(q2_hidden, 1, name="{}_q2".format(name))
                q2_heads[name] = _q2

        q1 = tf.reduce_mean(list(q1_heads.values()))
        q2 = tf.reduce_mean(list(q2_heads.values()))
        return q1_heads, q2_heads, q1, q2


class SACModel(LearningModel):
    def __init__(
        self,
        brain,
        lr=1e-4,
        h_size=128,
        epsilon=0.2,
        beta=1e-3,
        max_step=5e6,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        m_size=None,
        seed=0,
        stream_names=None,
    ):
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
        self.tau = 0.005
        if stream_names is None:
            stream_names = []
        LearningModel.__init__(
            self, m_size, normalize, use_recurrent, brain, seed, stream_names
        )
        if num_layers < 1:
            num_layers = 1
        self.last_reward, self.new_reward, self.update_reward = (
            self.create_reward_encoder()
        )
        if brain.vector_action_space_type == "continuous":
            self.policy_network = SACNetwork(
                brain = brain,
                m_size = m_size,
                h_size = h_size,
                normalize = normalize,
                use_recurrent = use_recurrent,
                num_layers = num_layers,
                seed = seed,
                stream_names=stream_names,
                is_target=False,
            )
            self.target_network = SACNetwork(
                brain = brain,
                m_size = m_size,
                h_size = h_size,
                normalize = normalize,
                use_recurrent = use_recurrent,
                num_layers = num_layers,
                seed = seed,
                stream_names=stream_names,
                is_target=True,
            )
            self.action_holder = self.policy_network.action_holder
        else:
            self.create_dc_actor_critic(h_size, num_layers)
        self.create_losses(
            self.policy_network.q1_heads,
            self.policy_network.q2_heads,
            beta,
            epsilon,
            lr,
            max_step,
            stream_names,
        )
        if normalize:
            self.update_normalization = self.policy_network.update_normalization
        self.create_inputs()

    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(
            0, name="last_reward", trainable=False, dtype=tf.float32
        )
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name="new_reward")
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    def create_inputs(self):
        self.vector_in = self.policy_network.vector_in
        self.visual_in = self.policy_network.visual_in
        self.next_vector_in = self.target_network.vector_in
        self.next_visual_in = self.target_network.visual_in

        self.output = self.policy_network.output
        self.value = self.policy_network.value
        self.all_log_probs = self.policy_network.all_log_probs

    def create_losses(
        self,
        q1_streams,
        q2_streams,
        beta,
        epsilon,
        lr,
        max_step,
        stream_names,
    ):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value_streams: Current value estimates from each value stream
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """
        self.min_policy_q = tf.minimum(self.policy_network.q1_p, self.policy_network.q2_p)

        self.target_entropy = -np.prod(self.act_size[0]).astype(np.float32)

        self.rewards_holders = []
        self.old_values = []
        for i in range(len(q1_streams)):
            returns_holder = tf.placeholder(
                shape=[None],
                dtype=tf.float32,
                name="{}_rewards".format(stream_names[i]),
            )
            # old_value = tf.placeholder(
            #     shape=[None],
            #     dtype=tf.float32,
            #     name="{}_value_estimate".format(stream_names[i]),
            # )
            self.rewards_holders.append(returns_holder)
            # self.old_values.append(old_value)
        self.advantage = tf.placeholder(
            shape=[None, 1], dtype=tf.float32, name="advantages"
        )
        self.learning_rate = tf.train.polynomial_decay(
            lr, self.global_step, max_step, 1e-10, power=1.0
        )

        decay_epsilon = tf.train.polynomial_decay(
            epsilon, self.global_step, max_step, 0.1, power=1.0
        )
        decay_beta = tf.train.polynomial_decay(
            beta, self.global_step, max_step, 1e-5, power=1.0
        )

        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(stream_names):
            q1_loss = 0.5 * tf.squared_difference(
                self.rewards_holders[i], tf.reduce_sum(q1_streams[name], axis=1)
            )
            q1_losses.append(q1_loss)
            q2_loss = 0.5 * tf.squared_difference(
                self.rewards_holders[i], tf.reduce_sum(q2_streams[name], axis=1)
            )
            q2_losses.append(q2_loss)
            # clipped_value_estimate = self.old_values[i] + tf.clip_by_value(
            #     tf.reduce_sum(value_streams[name], axis=1) - self.old_values[i],
            #     -decay_epsilon,
            #     decay_epsilon,
            # )
            # v_opt_a = tf.squared_difference(
            #     self.rewards_holders[i], tf.reduce_sum(value_streams[name], axis=1)
            # )
            # v_opt_b = tf.squared_difference(
            #     self.rewards_holders[i], clipped_value_estimate
            # )
        self.q1_loss = tf.reduce_mean(q1_losses)
        self.q2_loss = tf.reduce_mean(q2_losses)

        # Learn entropy coefficient
        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(1.0).astype(np.float32))
        self.ent_coef = tf.exp(self.log_ent_coef)

        self.entropy_loss = -tf.reduce_mean(
            self.log_ent_coef * tf.stop_gradient(self.policy_network.all_log_probs + self.target_entropy))

        self.policy_loss = tf.reduce_mean(self.ent_coef * self.policy_network.all_log_probs - self.policy_network.q1_p)

        # Only one value head, only one value loss
        v_backup = tf.stop_gradient(self.min_policy_q - self.ent_coef * self.policy_network.all_log_probs)
        self.value_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.policy_network.value, v_backup))

        self.total_value_loss = self.q1_loss + self.q2_loss + self.value_loss

    def create_sac_optimizers(self):
        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.target_update_op = [
                tf.assign(target, (1 - self.tau) * target + self.tau * source)
                for target, source in zip(self.target_network.value_vars, self.policy_network.value_vars)
            ]

        self.target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(self.target_network.value_vars, self.policy_network.value_vars)
                    ]

        self.update_batch_policy = policy_optimizer.minimize(self.policy_loss, var_list=self.policy_network.policy_vars)

        # Make sure policy is updated first, then value, then entropy.
        with tf.control_dependencies([self.update_batch_policy]):
            self.update_batch_value = value_optimizer.minimize(self.total_value_loss, var_list=self.policy_network.critic_vars)

            # Add entropy coefficient optimization operation if needed
            with tf.control_dependencies([self.update_batch_value]):
                self.update_batch_entropy = entropy_optimizer.minimize(self.entropy_loss, var_list=self.log_ent_coef)




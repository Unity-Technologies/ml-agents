import logging
import numpy as np
from typing import Dict, List, Optional

from mlagents.tf_utils import tf

from mlagents.trainers.models import LearningModel, LearningRateSchedule, EncoderType

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6  # Small value to avoid divide by zero
DISCRETE_TARGET_ENTROPY_SCALE = 0.2  # Roughly equal to e-greedy 0.05
CONTINUOUS_TARGET_ENTROPY_SCALE = 1.0  # TODO: Make these an optional hyperparam.

LOGGER = logging.getLogger("mlagents.trainers")

POLICY_SCOPE = ""
TARGET_SCOPE = "target_network"


class SACNetwork(LearningModel):
    """
    Base class for an SAC network. Implements methods for creating the actor and critic heads.
    """

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
        vis_encode_type=EncoderType.SIMPLE,
    ):
        LearningModel.__init__(
            self, m_size, normalize, use_recurrent, brain, seed, stream_names
        )
        self.normalize = normalize
        self.use_recurrent = use_recurrent
        self.num_layers = num_layers
        self.stream_names = stream_names
        self.h_size = h_size
        self.activ_fn = self.swish

        self.policy_memory_in: Optional[tf.Tensor] = None
        self.policy_memory_out: Optional[tf.Tensor] = None
        self.value_memory_in: Optional[tf.Tensor] = None
        self.value_memory_out: Optional[tf.Tensor] = None
        self.q1: Optional[tf.Tensor] = None
        self.q2: Optional[tf.Tensor] = None
        self.q1_p: Optional[tf.Tensor] = None
        self.q2_p: Optional[tf.Tensor] = None
        self.q1_memory_in: Optional[tf.Tensor] = None
        self.q2_memory_in: Optional[tf.Tensor] = None
        self.q1_memory_out: Optional[tf.Tensor] = None
        self.q2_memory_out: Optional[tf.Tensor] = None
        self.prev_action: Optional[tf.Tensor] = None
        self.action_masks: Optional[tf.Tensor] = None
        self.external_action_in: Optional[tf.Tensor] = None
        self.log_sigma_sq: Optional[tf.Tensor] = None
        self.entropy: Optional[tf.Tensor] = None
        self.deterministic_output: Optional[tf.Tensor] = None
        self.normalized_logprobs: Optional[tf.Tensor] = None
        self.action_probs: Optional[tf.Tensor] = None
        self.output_oh: Optional[tf.Tensor] = None
        self.output_pre: Optional[tf.Tensor] = None

        self.value_vars = None
        self.q_vars = None
        self.critic_vars = None
        self.policy_vars = None

        self.q1_heads: Optional[Dict[str, tf.Tensor]] = None
        self.q2_heads: Optional[Dict[str, tf.Tensor]] = None
        self.q1_pheads: Optional[Dict[str, tf.Tensor]] = None
        self.q2_pheads: Optional[Dict[str, tf.Tensor]] = None

    def get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def join_scopes(self, scope_1, scope_2):
        """
        Joins two scopes. Does so safetly (i.e., if one of the two scopes doesn't
        exist, don't add any backslashes)
        """
        if not scope_1:
            return scope_2
        if not scope_2:
            return scope_1
        else:
            return "/".join(filter(None, [scope_1, scope_2]))

    def create_cc_critic(self, hidden_value, scope, create_qs=True):
        """
        Creates just the critic network
        """
        scope = self.join_scopes(scope, "critic")
        self.create_sac_value_head(
            self.stream_names,
            hidden_value,
            self.num_layers,
            self.h_size,
            self.join_scopes(scope, "value"),
        )

        self.value_vars = self.get_vars(self.join_scopes(scope, "value"))

        if create_qs:
            hidden_q = tf.concat([hidden_value, self.external_action_in], axis=-1)
            hidden_qp = tf.concat([hidden_value, self.output], axis=-1)
            self.q1_heads, self.q2_heads, self.q1, self.q2 = self.create_q_heads(
                self.stream_names,
                hidden_q,
                self.num_layers,
                self.h_size,
                self.join_scopes(scope, "q"),
            )
            self.q1_pheads, self.q2_pheads, self.q1_p, self.q2_p = self.create_q_heads(
                self.stream_names,
                hidden_qp,
                self.num_layers,
                self.h_size,
                self.join_scopes(scope, "q"),
                reuse=True,
            )
            self.q_vars = self.get_vars(self.join_scopes(scope, "q"))
        self.critic_vars = self.get_vars(scope)

    def create_dc_critic(self, hidden_value, scope, create_qs=True):
        """
        Creates just the critic network
        """
        scope = self.join_scopes(scope, "critic")
        self.create_sac_value_head(
            self.stream_names,
            hidden_value,
            self.num_layers,
            self.h_size,
            self.join_scopes(scope, "value"),
        )

        self.value_vars = self.get_vars("/".join([scope, "value"]))

        if create_qs:
            self.q1_heads, self.q2_heads, self.q1, self.q2 = self.create_q_heads(
                self.stream_names,
                hidden_value,
                self.num_layers,
                self.h_size,
                self.join_scopes(scope, "q"),
                num_outputs=sum(self.act_size),
            )
            self.q1_pheads, self.q2_pheads, self.q1_p, self.q2_p = self.create_q_heads(
                self.stream_names,
                hidden_value,
                self.num_layers,
                self.h_size,
                self.join_scopes(scope, "q"),
                reuse=True,
                num_outputs=sum(self.act_size),
            )
            self.q_vars = self.get_vars(scope)
        self.critic_vars = self.get_vars(scope)

    def create_cc_actor(self, hidden_policy, scope):
        """
        Creates Continuous control actor for SAC.
        :param hidden_policy: Output of feature extractor (i.e. the input for vector obs, output of CNN for visual obs).
        :param num_layers: TF scope to assign whatever is created in this block.
        """
        # Create action input (continuous)
        self.action_holder = tf.placeholder(
            shape=[None, self.act_size[0]], dtype=tf.float32, name="action_holder"
        )
        self.external_action_in = self.action_holder

        scope = self.join_scopes(scope, "policy")

        with tf.variable_scope(scope):
            hidden_policy = self.create_vector_observation_encoder(
                hidden_policy,
                self.h_size,
                self.activ_fn,
                self.num_layers,
                "encoder",
                False,
            )
        if self.use_recurrent:
            hidden_policy, memory_out = self.create_recurrent_encoder(
                hidden_policy,
                self.policy_memory_in,
                self.sequence_length,
                name="lstm_policy",
            )
            self.policy_memory_out = memory_out
        with tf.variable_scope(scope):
            mu = tf.layers.dense(
                hidden_policy,
                self.act_size[0],
                activation=None,
                name="mu",
                kernel_initializer=LearningModel.scaled_init(0.01),
            )

            # Policy-dependent log_sigma_sq
            log_sigma_sq = tf.layers.dense(
                hidden_policy,
                self.act_size[0],
                activation=None,
                name="log_std",
                kernel_initializer=LearningModel.scaled_init(0.01),
            )

            self.log_sigma_sq = tf.clip_by_value(log_sigma_sq, LOG_STD_MIN, LOG_STD_MAX)

            sigma_sq = tf.exp(self.log_sigma_sq)

            # Do the reparameterization trick
            policy_ = mu + tf.random_normal(tf.shape(mu)) * sigma_sq

            _gauss_pre = -0.5 * (
                ((policy_ - mu) / (tf.exp(self.log_sigma_sq) + EPSILON)) ** 2
                + 2 * self.log_sigma_sq
                + np.log(2 * np.pi)
            )

            all_probs = tf.reduce_sum(_gauss_pre, axis=1, keepdims=True)

            self.entropy = tf.reduce_sum(
                self.log_sigma_sq + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1
            )

            # Squash probabilities
            # Keep deterministic around in case we want to use it.
            self.deterministic_output = tf.tanh(mu)

            # Note that this is just for symmetry with PPO.
            self.output_pre = tf.tanh(policy_)

            # Squash correction
            all_probs -= tf.reduce_sum(
                tf.log(1 - self.output_pre ** 2 + EPSILON), axis=1, keepdims=True
            )

            self.all_log_probs = all_probs
            self.selected_actions = tf.stop_gradient(self.output_pre)

            self.action_probs = all_probs

        # Extract output for Barracuda
        self.output = tf.identity(self.output_pre, name="action")

        # Get all policy vars
        self.policy_vars = self.get_vars(scope)

    def create_dc_actor(self, hidden_policy, scope):
        """
        Creates Discrete control actor for SAC.
        :param hidden_policy: Output of feature extractor (i.e. the input for vector obs, output of CNN for visual obs).
        :param num_layers: TF scope to assign whatever is created in this block.
        """
        scope = self.join_scopes(scope, "policy")

        # Create inputs outside of the scope
        self.action_masks = tf.placeholder(
            shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
        )

        if self.use_recurrent:
            self.prev_action = tf.placeholder(
                shape=[None, len(self.act_size)], dtype=tf.int32, name="prev_action"
            )

        with tf.variable_scope(scope):
            hidden_policy = self.create_vector_observation_encoder(
                hidden_policy,
                self.h_size,
                self.activ_fn,
                self.num_layers,
                "encoder",
                False,
            )
        if self.use_recurrent:
            prev_action_oh = tf.concat(
                [
                    tf.one_hot(self.prev_action[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            hidden_policy = tf.concat([hidden_policy, prev_action_oh], axis=1)

            hidden_policy, memory_out = self.create_recurrent_encoder(
                hidden_policy,
                self.policy_memory_in,
                self.sequence_length,
                name="lstm_policy",
            )
            self.policy_memory_out = memory_out
        with tf.variable_scope(scope):
            policy_branches = []
            for size in self.act_size:
                policy_branches.append(
                    tf.layers.dense(
                        hidden_policy,
                        size,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=tf.initializers.variance_scaling(0.01),
                    )
                )
            all_logits = tf.concat(policy_branches, axis=1, name="action_probs")

            output, normalized_probs, normalized_logprobs = self.create_discrete_action_masking_layer(
                all_logits, self.action_masks, self.act_size
            )

            self.action_probs = normalized_probs

            # Really, this is entropy, but it has an analogous purpose to the log probs in the
            # continuous case.
            self.all_log_probs = self.action_probs * normalized_logprobs
            self.output = output

            # Create action input (discrete)
            self.action_holder = tf.placeholder(
                shape=[None, len(policy_branches)], dtype=tf.int32, name="action_holder"
            )

            self.output_oh = tf.concat(
                [
                    tf.one_hot(self.action_holder[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )

            # For Curiosity and GAIL to retrieve selected actions. We don't
            # need the mask at this point because it's already stored in the buffer.
            self.selected_actions = tf.stop_gradient(self.output_oh)

            self.external_action_in = tf.concat(
                [
                    tf.one_hot(self.action_holder[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )

            # This is total entropy over all branches
            self.entropy = -1 * tf.reduce_sum(self.all_log_probs, axis=1)

        # Extract the normalized logprobs for Barracuda
        self.normalized_logprobs = tf.identity(normalized_logprobs, name="action")

        # We kept the LSTMs at a different scope than the rest, so add them if they exist.
        self.policy_vars = self.get_vars(scope)
        if self.use_recurrent:
            self.policy_vars += self.get_vars("lstm")

    def create_sac_value_head(
        self, stream_names, hidden_input, num_layers, h_size, scope
    ):
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        :param num_layers: Number of hidden layers for value network
        :param h_size: size of hidden layers for value network
        :param scope: TF scope for value network.
        """
        with tf.variable_scope(scope):
            value_hidden = self.create_vector_observation_encoder(
                hidden_input, h_size, self.activ_fn, num_layers, "encoder", False
            )
            if self.use_recurrent:
                value_hidden, memory_out = self.create_recurrent_encoder(
                    value_hidden,
                    self.value_memory_in,
                    self.sequence_length,
                    name="lstm_value",
                )
                self.value_memory_out = memory_out
            self.create_value_heads(stream_names, value_hidden)

    def create_q_heads(
        self,
        stream_names,
        hidden_input,
        num_layers,
        h_size,
        scope,
        reuse=False,
        num_outputs=1,
    ):
        """
        Creates two q heads for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        :param num_layers: Number of hidden layers for Q network
        :param h_size: size of hidden layers for Q network
        :param scope: TF scope for Q network.
        :param reuse: Whether or not to reuse variables. Useful for creating Q of policy.
        :param num_outputs: Number of outputs of each Q function. If discrete, equal to number of actions.
        """
        with tf.variable_scope(self.join_scopes(scope, "q1_encoding"), reuse=reuse):
            q1_hidden = self.create_vector_observation_encoder(
                hidden_input, h_size, self.activ_fn, num_layers, "q1_encoder", reuse
            )
            if self.use_recurrent:
                q1_hidden, memory_out = self.create_recurrent_encoder(
                    q1_hidden, self.q1_memory_in, self.sequence_length, name="lstm_q1"
                )
                self.q1_memory_out = memory_out

            q1_heads = {}
            for name in stream_names:
                _q1 = tf.layers.dense(q1_hidden, num_outputs, name="{}_q1".format(name))
                q1_heads[name] = _q1

            q1 = tf.reduce_mean(list(q1_heads.values()), axis=0)
        with tf.variable_scope(self.join_scopes(scope, "q2_encoding"), reuse=reuse):
            q2_hidden = self.create_vector_observation_encoder(
                hidden_input, h_size, self.activ_fn, num_layers, "q2_encoder", reuse
            )
            if self.use_recurrent:
                q2_hidden, memory_out = self.create_recurrent_encoder(
                    q2_hidden, self.q2_memory_in, self.sequence_length, name="lstm_q2"
                )
                self.q2_memory_out = memory_out

            q2_heads = {}
            for name in stream_names:
                _q2 = tf.layers.dense(q2_hidden, num_outputs, name="{}_q2".format(name))
                q2_heads[name] = _q2

            q2 = tf.reduce_mean(list(q2_heads.values()), axis=0)

        return q1_heads, q2_heads, q1, q2

    def copy_normalization(self, mean, variance, steps):
        """
        Copies the mean, variance, and steps into the normalizers of the
        input of this SACNetwork. Used to copy the normalizer from the policy network
        to the target network.
        param mean: Tensor containing the mean.
        param variance: Tensor containing the variance
        param steps: Tensor containing the number of steps.
        """
        update_mean = tf.assign(self.running_mean, mean)
        update_variance = tf.assign(self.running_variance, variance)
        update_norm_step = tf.assign(self.normalization_steps, steps)
        return tf.group([update_mean, update_variance, update_norm_step])


class SACTargetNetwork(SACNetwork):
    """
    Instantiation for the SAC target network. Only contains a single
    value estimator and is updated from the Policy Network.
    """

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
        vis_encode_type=EncoderType.SIMPLE,
    ):
        super().__init__(
            brain,
            m_size,
            h_size,
            normalize,
            use_recurrent,
            num_layers,
            stream_names,
            seed,
            vis_encode_type,
        )
        if self.use_recurrent:
            self.memory_in = tf.placeholder(
                shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
            )
            self.value_memory_in = self.memory_in
        with tf.variable_scope(TARGET_SCOPE):
            hidden_streams = self.create_observation_streams(
                1,
                self.h_size,
                0,
                vis_encode_type=vis_encode_type,
                stream_scopes=["critic/value/"],
            )
        if brain.vector_action_space_type == "continuous":
            self.create_cc_critic(hidden_streams[0], TARGET_SCOPE, create_qs=False)
        else:
            self.create_dc_critic(hidden_streams[0], TARGET_SCOPE, create_qs=False)
        if self.use_recurrent:
            self.memory_out = tf.concat(
                self.value_memory_out, axis=1
            )  # Needed for Barracuda to work


class SACPolicyNetwork(SACNetwork):
    """
    Instantiation for SAC policy network. Contains a dual Q estimator,
    a value estimator, and the actual policy network.
    """

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
        vis_encode_type=EncoderType.SIMPLE,
    ):
        super().__init__(
            brain,
            m_size,
            h_size,
            normalize,
            use_recurrent,
            num_layers,
            stream_names,
            seed,
            vis_encode_type,
        )
        self.share_ac_cnn = False
        if self.use_recurrent:
            self.create_memory_ins(self.m_size)

        hidden_policy, hidden_critic = self.create_observation_ins(
            vis_encode_type, self.share_ac_cnn
        )

        if brain.vector_action_space_type == "continuous":
            self.create_cc_actor(hidden_policy, POLICY_SCOPE)
            self.create_cc_critic(hidden_critic, POLICY_SCOPE)

        else:
            self.create_dc_actor(hidden_policy, POLICY_SCOPE)
            self.create_dc_critic(hidden_critic, POLICY_SCOPE)

        if self.share_ac_cnn:
            # Make sure that the policy also contains the CNN
            self.policy_vars += self.get_vars(
                self.join_scopes(POLICY_SCOPE, "critic/value/main_graph_0_encoder0")
            )
        if self.use_recurrent:
            mem_outs = [
                self.value_memory_out,
                self.q1_memory_out,
                self.q2_memory_out,
                self.policy_memory_out,
            ]
            self.memory_out = tf.concat(mem_outs, axis=1)

    def create_memory_ins(self, m_size):
        """
        Creates the memory input placeholders for LSTM.
        :param m_size: the total size of the memory.
        """
        # Create the Policy input separate from the rest
        # This is so in inference we only have to run the Policy network.
        # Barracuda will grab the recurrent_in and recurrent_out named tensors.
        self.inference_memory_in = tf.placeholder(
            shape=[None, m_size // 4], dtype=tf.float32, name="recurrent_in"
        )
        # We assume m_size is divisible by 4
        # Create the non-Policy inputs
        # Use a default placeholder here so nothing has to be provided during
        # Barracuda inference. Note that the default value is just the tiled input
        # for the policy, which is thrown away.
        three_fourths_m_size = m_size * 3 // 4
        self.other_memory_in = tf.placeholder_with_default(
            input=tf.tile(self.inference_memory_in, [1, 3]),
            shape=[None, three_fourths_m_size],
            name="other_recurrent_in",
        )

        # Concat and use this as the "placeholder"
        # for training
        self.memory_in = tf.concat(
            [self.other_memory_in, self.inference_memory_in], axis=1
        )

        # Re-break-up for each network
        num_mems = 4
        mem_ins = []
        for i in range(num_mems):
            _start = m_size // num_mems * i
            _end = m_size // num_mems * (i + 1)
            mem_ins.append(self.memory_in[:, _start:_end])
        self.value_memory_in = mem_ins[0]
        self.q1_memory_in = mem_ins[1]
        self.q2_memory_in = mem_ins[2]
        self.policy_memory_in = mem_ins[3]

    def create_observation_ins(self, vis_encode_type, share_ac_cnn):
        """
        Creates the observation inputs, and a CNN if needed,
        :param vis_encode_type: Type of CNN encoder.
        :param share_ac_cnn: Whether or not to share the actor and critic CNNs.
        :return A tuple of (hidden_policy, hidden_critic). We don't save it to self since they're used
        once and thrown away.
        """
        if share_ac_cnn:
            with tf.variable_scope(POLICY_SCOPE):
                hidden_streams = self.create_observation_streams(
                    1,
                    self.h_size,
                    0,
                    vis_encode_type=vis_encode_type,
                    stream_scopes=["critic/value/"],
                )
            hidden_policy = hidden_streams[0]
            hidden_critic = hidden_streams[0]
        else:
            with tf.variable_scope(POLICY_SCOPE):
                hidden_streams = self.create_observation_streams(
                    2,
                    self.h_size,
                    0,
                    vis_encode_type=vis_encode_type,
                    stream_scopes=["policy/", "critic/value/"],
                )
            hidden_policy = hidden_streams[0]
            hidden_critic = hidden_streams[1]
        return hidden_policy, hidden_critic


class SACModel(LearningModel):
    def __init__(
        self,
        brain,
        lr=1e-4,
        lr_schedule=LearningRateSchedule.CONSTANT,
        h_size=128,
        init_entcoef=0.1,
        max_step=5e6,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        m_size=None,
        seed=0,
        stream_names=None,
        tau=0.005,
        gammas=None,
        vis_encode_type=EncoderType.SIMPLE,
    ):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: BrainInfo used to generate specific network graph.
        :param lr: Learning rate.
        :param lr_schedule: Learning rate decay schedule.
        :param h_size: Size of hidden layers
        :param init_entcoef: Initial value for entropy coefficient. Set lower to learn faster,
            set higher to explore more.
        :return: a sub-class of PPOAgent tailored to the environment.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers: Number of hidden layers between encoded input and policy & value layers
        :param tau: Strength of soft-Q update.
        :param m_size: Size of brain memory.
        """
        self.tau = tau
        self.gammas = gammas
        self.brain = brain
        self.init_entcoef = init_entcoef
        if stream_names is None:
            stream_names = []
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.use_dones_in_backup = {name: tf.Variable(1.0) for name in stream_names}
        self.disable_use_dones = {
            name: self.use_dones_in_backup[name].assign(0.0) for name in stream_names
        }
        LearningModel.__init__(
            self, m_size, normalize, use_recurrent, brain, seed, stream_names
        )
        if num_layers < 1:
            num_layers = 1

        self.target_init_op: List[tf.Tensor] = []
        self.target_update_op: List[tf.Tensor] = []
        self.update_batch_policy: Optional[tf.Operation] = None
        self.update_batch_value: Optional[tf.Operation] = None
        self.update_batch_entropy: Optional[tf.Operation] = None

        self.policy_network = SACPolicyNetwork(
            brain=brain,
            m_size=m_size,
            h_size=h_size,
            normalize=normalize,
            use_recurrent=use_recurrent,
            num_layers=num_layers,
            seed=seed,
            stream_names=stream_names,
            vis_encode_type=vis_encode_type,
        )
        self.target_network = SACTargetNetwork(
            brain=brain,
            m_size=m_size // 4 if m_size else None,
            h_size=h_size,
            normalize=normalize,
            use_recurrent=use_recurrent,
            num_layers=num_layers,
            seed=seed,
            stream_names=stream_names,
            vis_encode_type=vis_encode_type,
        )
        self.create_inputs_and_outputs()
        self.learning_rate = self.create_learning_rate(
            lr_schedule, lr, self.global_step, max_step
        )
        self.create_losses(
            self.policy_network.q1_heads,
            self.policy_network.q2_heads,
            lr,
            max_step,
            stream_names,
            discrete=self.brain.vector_action_space_type == "discrete",
        )

        self.selected_actions = (
            self.policy_network.selected_actions
        )  # For GAIL and other reward signals
        if normalize:
            target_update_norm = self.target_network.copy_normalization(
                self.policy_network.running_mean,
                self.policy_network.running_variance,
                self.policy_network.normalization_steps,
            )
            self.update_normalization = tf.group(
                [self.policy_network.update_normalization, target_update_norm]
            )
            self.running_mean = self.policy_network.running_mean
            self.running_variance = self.policy_network.running_variance
            self.normalization_steps = self.policy_network.normalization_steps

    def create_inputs_and_outputs(self):
        """
        Assign the higher-level SACModel's inputs and outputs to those of its policy or
        target network.
        """
        self.vector_in = self.policy_network.vector_in
        self.visual_in = self.policy_network.visual_in
        self.next_vector_in = self.target_network.vector_in
        self.next_visual_in = self.target_network.visual_in
        self.action_holder = self.policy_network.action_holder
        self.sequence_length = self.policy_network.sequence_length
        self.next_sequence_length = self.target_network.sequence_length
        if self.brain.vector_action_space_type == "discrete":
            self.action_masks = self.policy_network.action_masks
        else:
            self.output_pre = self.policy_network.output_pre

        self.output = self.policy_network.output
        # Don't use value estimate during inference. TODO: Check why PPO uses value_estimate in inference.
        self.value = tf.identity(
            self.policy_network.value, name="value_estimate_unused"
        )
        self.value_heads = self.policy_network.value_heads
        self.all_log_probs = self.policy_network.all_log_probs
        self.dones_holder = tf.placeholder(
            shape=[None], dtype=tf.float32, name="dones_holder"
        )
        # This is just a dummy to get BC to work. PPO has this but SAC doesn't.
        # TODO: Proper input and output specs for models
        self.epsilon = tf.placeholder(
            shape=[None, self.act_size[0]], dtype=tf.float32, name="epsilon"
        )
        if self.use_recurrent:
            self.memory_in = self.policy_network.memory_in
            self.memory_out = self.policy_network.memory_out

            # For Barracuda
            self.inference_memory_out = tf.identity(
                self.policy_network.policy_memory_out, name="recurrent_out"
            )

            if self.brain.vector_action_space_type == "discrete":
                self.prev_action = self.policy_network.prev_action
            self.next_memory_in = self.target_network.memory_in

    def create_losses(
        self, q1_streams, q2_streams, lr, max_step, stream_names, discrete=False
    ):
        """
        Creates training-specific Tensorflow ops for SAC models.
        :param q1_streams: Q1 streams from policy network
        :param q1_streams: Q2 streams from policy network
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        :param stream_names: List of reward stream names.
        :param discrete: Whether or not to use discrete action losses.
        """

        if discrete:
            self.target_entropy = [
                DISCRETE_TARGET_ENTROPY_SCALE * np.log(i).astype(np.float32)
                for i in self.act_size
            ]
        else:
            self.target_entropy = (
                -1
                * CONTINUOUS_TARGET_ENTROPY_SCALE
                * np.prod(self.act_size[0]).astype(np.float32)
            )

        self.rewards_holders = {}
        self.min_policy_qs = {}

        for name in stream_names:
            if discrete:
                _branched_mpq1 = self.apply_as_branches(
                    self.policy_network.q1_pheads[name]
                    * self.policy_network.action_probs
                )
                branched_mpq1 = tf.stack(
                    [
                        tf.reduce_sum(_br, axis=1, keep_dims=True)
                        for _br in _branched_mpq1
                    ]
                )
                _q1_p_mean = tf.reduce_mean(branched_mpq1, axis=0)

                _branched_mpq2 = self.apply_as_branches(
                    self.policy_network.q2_pheads[name]
                    * self.policy_network.action_probs
                )
                branched_mpq2 = tf.stack(
                    [
                        tf.reduce_sum(_br, axis=1, keep_dims=True)
                        for _br in _branched_mpq2
                    ]
                )
                _q2_p_mean = tf.reduce_mean(branched_mpq2, axis=0)

                self.min_policy_qs[name] = tf.minimum(_q1_p_mean, _q2_p_mean)
            else:
                self.min_policy_qs[name] = tf.minimum(
                    self.policy_network.q1_pheads[name],
                    self.policy_network.q2_pheads[name],
                )

            rewards_holder = tf.placeholder(
                shape=[None], dtype=tf.float32, name="{}_rewards".format(name)
            )
            self.rewards_holders[name] = rewards_holder

        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        expanded_dones = tf.expand_dims(self.dones_holder, axis=-1)
        for i, name in enumerate(stream_names):
            _expanded_rewards = tf.expand_dims(self.rewards_holders[name], axis=-1)

            q_backup = tf.stop_gradient(
                _expanded_rewards
                + (1.0 - self.use_dones_in_backup[name] * expanded_dones)
                * self.gammas[i]
                * self.target_network.value_heads[name]
            )

            if discrete:
                # We need to break up the Q functions by branch, and update them individually.
                branched_q1_stream = self.apply_as_branches(
                    self.policy_network.external_action_in * q1_streams[name]
                )
                branched_q2_stream = self.apply_as_branches(
                    self.policy_network.external_action_in * q2_streams[name]
                )

                # Reduce each branch into scalar
                branched_q1_stream = [
                    tf.reduce_sum(_branch, axis=1, keep_dims=True)
                    for _branch in branched_q1_stream
                ]
                branched_q2_stream = [
                    tf.reduce_sum(_branch, axis=1, keep_dims=True)
                    for _branch in branched_q2_stream
                ]

                q1_stream = tf.reduce_mean(branched_q1_stream, axis=0)
                q2_stream = tf.reduce_mean(branched_q2_stream, axis=0)

            else:
                q1_stream = q1_streams[name]
                q2_stream = q2_streams[name]

            _q1_loss = 0.5 * tf.reduce_mean(
                tf.to_float(self.mask) * tf.squared_difference(q_backup, q1_stream)
            )

            _q2_loss = 0.5 * tf.reduce_mean(
                tf.to_float(self.mask) * tf.squared_difference(q_backup, q2_stream)
            )

            q1_losses.append(_q1_loss)
            q2_losses.append(_q2_loss)

        self.q1_loss = tf.reduce_mean(q1_losses)
        self.q2_loss = tf.reduce_mean(q2_losses)

        # Learn entropy coefficient
        if discrete:
            # Create a log_ent_coef for each branch
            self.log_ent_coef = tf.get_variable(
                "log_ent_coef",
                dtype=tf.float32,
                initializer=np.log([self.init_entcoef] * len(self.act_size)).astype(
                    np.float32
                ),
                trainable=True,
            )
        else:
            self.log_ent_coef = tf.get_variable(
                "log_ent_coef",
                dtype=tf.float32,
                initializer=np.log(self.init_entcoef).astype(np.float32),
                trainable=True,
            )

        self.ent_coef = tf.exp(self.log_ent_coef)
        if discrete:
            # We also have to do a different entropy and target_entropy per branch.
            branched_log_probs = self.apply_as_branches(
                self.policy_network.all_log_probs
            )
            branched_ent_sums = tf.stack(
                [
                    tf.reduce_sum(_lp, axis=1, keep_dims=True) + _te
                    for _lp, _te in zip(branched_log_probs, self.target_entropy)
                ],
                axis=1,
            )
            self.entropy_loss = -tf.reduce_mean(
                tf.to_float(self.mask)
                * tf.reduce_mean(
                    self.log_ent_coef
                    * tf.squeeze(tf.stop_gradient(branched_ent_sums), axis=2),
                    axis=1,
                )
            )

            # Same with policy loss, we have to do the loss per branch and average them,
            # so that larger branches don't get more weight.
            # The equivalent KL divergence from Eq 10 of Haarnoja et al. is also pi*log(pi) - Q
            branched_q_term = self.apply_as_branches(
                self.policy_network.action_probs * self.policy_network.q1_p
            )

            branched_policy_loss = tf.stack(
                [
                    tf.reduce_sum(self.ent_coef[i] * _lp - _qt, axis=1, keep_dims=True)
                    for i, (_lp, _qt) in enumerate(
                        zip(branched_log_probs, branched_q_term)
                    )
                ]
            )
            self.policy_loss = tf.reduce_mean(
                tf.to_float(self.mask) * tf.squeeze(branched_policy_loss)
            )

            # Do vbackup entropy bonus per branch as well.
            branched_ent_bonus = tf.stack(
                [
                    tf.reduce_sum(self.ent_coef[i] * _lp, axis=1, keep_dims=True)
                    for i, _lp in enumerate(branched_log_probs)
                ]
            )
            value_losses = []
            for name in stream_names:
                v_backup = tf.stop_gradient(
                    self.min_policy_qs[name]
                    - tf.reduce_mean(branched_ent_bonus, axis=0)
                )
                value_losses.append(
                    0.5
                    * tf.reduce_mean(
                        tf.to_float(self.mask)
                        * tf.squared_difference(
                            self.policy_network.value_heads[name], v_backup
                        )
                    )
                )

        else:
            self.entropy_loss = -tf.reduce_mean(
                self.log_ent_coef
                * tf.to_float(self.mask)
                * tf.stop_gradient(
                    tf.reduce_sum(
                        self.policy_network.all_log_probs + self.target_entropy,
                        axis=1,
                        keep_dims=True,
                    )
                )
            )
            batch_policy_loss = tf.reduce_mean(
                self.ent_coef * self.policy_network.all_log_probs
                - self.policy_network.q1_p,
                axis=1,
            )
            self.policy_loss = tf.reduce_mean(
                tf.to_float(self.mask) * batch_policy_loss
            )

            value_losses = []
            for name in stream_names:
                v_backup = tf.stop_gradient(
                    self.min_policy_qs[name]
                    - tf.reduce_sum(
                        self.ent_coef * self.policy_network.all_log_probs, axis=1
                    )
                )
                value_losses.append(
                    0.5
                    * tf.reduce_mean(
                        tf.to_float(self.mask)
                        * tf.squared_difference(
                            self.policy_network.value_heads[name], v_backup
                        )
                    )
                )
        self.value_loss = tf.reduce_mean(value_losses)

        self.total_value_loss = self.q1_loss + self.q2_loss + self.value_loss

        self.entropy = self.policy_network.entropy

    def apply_as_branches(self, concat_logits):
        """
        Takes in a concatenated set of logits and breaks it up into a list of non-concatenated logits, one per
        action branch
        """
        action_idx = [0] + list(np.cumsum(self.act_size))
        branches_logits = [
            concat_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(self.act_size))
        ]
        return branches_logits

    def create_sac_optimizers(self):
        """
        Creates the Adam optimizers and update ops for SAC, including
        the policy, value, and entropy updates, as well as the target network update.
        """
        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.target_update_op = [
            tf.assign(target, (1 - self.tau) * target + self.tau * source)
            for target, source in zip(
                self.target_network.value_vars, self.policy_network.value_vars
            )
        ]
        LOGGER.debug("value_vars")
        self.print_all_vars(self.policy_network.value_vars)
        LOGGER.debug("targvalue_vars")
        self.print_all_vars(self.target_network.value_vars)
        LOGGER.debug("critic_vars")
        self.print_all_vars(self.policy_network.critic_vars)
        LOGGER.debug("q_vars")
        self.print_all_vars(self.policy_network.q_vars)
        LOGGER.debug("policy_vars")
        self.print_all_vars(self.policy_network.policy_vars)

        self.target_init_op = [
            tf.assign(target, source)
            for target, source in zip(
                self.target_network.value_vars, self.policy_network.value_vars
            )
        ]

        self.update_batch_policy = policy_optimizer.minimize(
            self.policy_loss, var_list=self.policy_network.policy_vars
        )

        # Make sure policy is updated first, then value, then entropy.
        with tf.control_dependencies([self.update_batch_policy]):
            self.update_batch_value = value_optimizer.minimize(
                self.total_value_loss, var_list=self.policy_network.critic_vars
            )
            # Add entropy coefficient optimization operation
            with tf.control_dependencies([self.update_batch_value]):
                self.update_batch_entropy = entropy_optimizer.minimize(
                    self.entropy_loss, var_list=self.log_ent_coef
                )

    def print_all_vars(self, variables):
        for _var in variables:
            LOGGER.debug(_var)

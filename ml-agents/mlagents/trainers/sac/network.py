from typing import Dict, Optional
from mlagents.tf_utils import tf
from mlagents.trainers.models import ModelUtils, EncoderType

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6  # Small value to avoid divide by zero
DISCRETE_TARGET_ENTROPY_SCALE = 0.2  # Roughly equal to e-greedy 0.05
CONTINUOUS_TARGET_ENTROPY_SCALE = 1.0  # TODO: Make these an optional hyperparam.
POLICY_SCOPE = ""
TARGET_SCOPE = "target_network"


class SACNetwork:
    """
    Base class for an SAC network. Implements methods for creating the actor and critic heads.
    """

    def __init__(
        self,
        policy=None,
        m_size=None,
        h_size=128,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        stream_names=None,
        vis_encode_type=EncoderType.SIMPLE,
    ):
        self.normalize = normalize
        self.use_recurrent = use_recurrent
        self.num_layers = num_layers
        self.stream_names = stream_names
        self.h_size = h_size
        self.activ_fn = ModelUtils.swish

        self.sequence_length_ph = tf.placeholder(
            shape=None, dtype=tf.int32, name="sac_sequence_length"
        )

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

        self.q1_heads: Dict[str, tf.Tensor] = None
        self.q2_heads: Dict[str, tf.Tensor] = None
        self.q1_pheads: Dict[str, tf.Tensor] = None
        self.q2_pheads: Dict[str, tf.Tensor] = None

        self.policy = policy

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

    def create_value_heads(self, stream_names, hidden_input):
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        self.value_heads = {}
        for name in stream_names:
            value = tf.layers.dense(hidden_input, 1, name="{}_value".format(name))
            self.value_heads[name] = value
        self.value = tf.reduce_mean(list(self.value_heads.values()), 0)

    def _create_cc_critic(self, hidden_value, scope, create_qs=True):
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
        self.external_action_in = tf.placeholder(
            shape=[None, self.policy.act_size[0]],
            dtype=tf.float32,
            name="external_action_in",
        )
        self.value_vars = self.get_vars(self.join_scopes(scope, "value"))
        if create_qs:
            hidden_q = tf.concat([hidden_value, self.external_action_in], axis=-1)
            hidden_qp = tf.concat([hidden_value, self.policy.output], axis=-1)
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

    def _create_dc_critic(self, hidden_value, scope, create_qs=True):
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
                num_outputs=sum(self.policy.act_size),
            )
            self.q1_pheads, self.q2_pheads, self.q1_p, self.q2_p = self.create_q_heads(
                self.stream_names,
                hidden_value,
                self.num_layers,
                self.h_size,
                self.join_scopes(scope, "q"),
                reuse=True,
                num_outputs=sum(self.policy.act_size),
            )
            self.q_vars = self.get_vars(scope)
        self.critic_vars = self.get_vars(scope)

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
            value_hidden = ModelUtils.create_vector_observation_encoder(
                hidden_input, h_size, self.activ_fn, num_layers, "encoder", False
            )
            if self.use_recurrent:
                value_hidden, memory_out = ModelUtils.create_recurrent_encoder(
                    value_hidden,
                    self.value_memory_in,
                    self.sequence_length_ph,
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
            q1_hidden = ModelUtils.create_vector_observation_encoder(
                hidden_input, h_size, self.activ_fn, num_layers, "q1_encoder", reuse
            )
            if self.use_recurrent:
                q1_hidden, memory_out = ModelUtils.create_recurrent_encoder(
                    q1_hidden,
                    self.q1_memory_in,
                    self.sequence_length_ph,
                    name="lstm_q1",
                )
                self.q1_memory_out = memory_out

            q1_heads = {}
            for name in stream_names:
                _q1 = tf.layers.dense(q1_hidden, num_outputs, name="{}_q1".format(name))
                q1_heads[name] = _q1

            q1 = tf.reduce_mean(list(q1_heads.values()), axis=0)
        with tf.variable_scope(self.join_scopes(scope, "q2_encoding"), reuse=reuse):
            q2_hidden = ModelUtils.create_vector_observation_encoder(
                hidden_input, h_size, self.activ_fn, num_layers, "q2_encoder", reuse
            )
            if self.use_recurrent:
                q2_hidden, memory_out = ModelUtils.create_recurrent_encoder(
                    q2_hidden,
                    self.q2_memory_in,
                    self.sequence_length_ph,
                    name="lstm_q2",
                )
                self.q2_memory_out = memory_out

            q2_heads = {}
            for name in stream_names:
                _q2 = tf.layers.dense(q2_hidden, num_outputs, name="{}_q2".format(name))
                q2_heads[name] = _q2

            q2 = tf.reduce_mean(list(q2_heads.values()), axis=0)

        return q1_heads, q2_heads, q1, q2


class SACTargetNetwork(SACNetwork):
    """
    Instantiation for the SAC target network. Only contains a single
    value estimator and is updated from the Policy Network.
    """

    def __init__(
        self,
        policy,
        m_size=None,
        h_size=128,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        stream_names=None,
        vis_encode_type=EncoderType.SIMPLE,
    ):
        super().__init__(
            policy,
            m_size,
            h_size,
            normalize,
            use_recurrent,
            num_layers,
            stream_names,
            vis_encode_type,
        )
        with tf.variable_scope(TARGET_SCOPE):
            self.visual_in = ModelUtils.create_visual_input_placeholders(
                policy.brain.camera_resolutions
            )
            self.vector_in = ModelUtils.create_vector_input(policy.vec_obs_size)
            if self.policy.normalize:
                normalization_tensors = ModelUtils.create_normalizer(self.vector_in)
                self.update_normalization_op = normalization_tensors.update_op
                self.normalization_steps = normalization_tensors.steps
                self.running_mean = normalization_tensors.running_mean
                self.running_variance = normalization_tensors.running_variance
                self.processed_vector_in = ModelUtils.normalize_vector_obs(
                    self.vector_in,
                    self.running_mean,
                    self.running_variance,
                    self.normalization_steps,
                )
            else:
                self.processed_vector_in = self.vector_in
                self.update_normalization_op = None

            if self.policy.use_recurrent:
                self.memory_in = tf.placeholder(
                    shape=[None, m_size], dtype=tf.float32, name="target_recurrent_in"
                )
                self.value_memory_in = self.memory_in
            hidden_streams = ModelUtils.create_observation_streams(
                self.visual_in,
                self.processed_vector_in,
                1,
                self.h_size,
                0,
                vis_encode_type=vis_encode_type,
                stream_scopes=["critic/value/"],
            )
        if self.policy.use_continuous_act:
            self._create_cc_critic(hidden_streams[0], TARGET_SCOPE, create_qs=False)
        else:
            self._create_dc_critic(hidden_streams[0], TARGET_SCOPE, create_qs=False)
        if self.use_recurrent:
            self.memory_out = tf.concat(
                self.value_memory_out, axis=1
            )  # Needed for Barracuda to work

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


class SACPolicyNetwork(SACNetwork):
    """
    Instantiation for SAC policy network. Contains a dual Q estimator,
    a value estimator, and a reference to the actual policy network.
    """

    def __init__(
        self,
        policy,
        m_size=None,
        h_size=128,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        stream_names=None,
        vis_encode_type=EncoderType.SIMPLE,
    ):
        super().__init__(
            policy,
            m_size,
            h_size,
            normalize,
            use_recurrent,
            num_layers,
            stream_names,
            vis_encode_type,
        )
        if self.policy.use_recurrent:
            self._create_memory_ins(m_size)

        hidden_critic = self._create_observation_in(vis_encode_type)
        self.policy.output = self.policy.output
        # Use the sequence length of the policy
        self.sequence_length_ph = self.policy.sequence_length_ph

        if self.policy.use_continuous_act:
            self._create_cc_critic(hidden_critic, POLICY_SCOPE)

        else:
            self._create_dc_critic(hidden_critic, POLICY_SCOPE)

        if self.use_recurrent:
            mem_outs = [self.value_memory_out, self.q1_memory_out, self.q2_memory_out]
            self.memory_out = tf.concat(mem_outs, axis=1)

    def _create_memory_ins(self, m_size):
        """
        Creates the memory input placeholders for LSTM.
        :param m_size: the total size of the memory.
        """
        self.memory_in = tf.placeholder(
            shape=[None, m_size * 3], dtype=tf.float32, name="value_recurrent_in"
        )

        # Re-break-up for each network
        num_mems = 3
        input_size = self.memory_in.get_shape().as_list()[1]
        mem_ins = []
        for i in range(num_mems):
            _start = input_size // num_mems * i
            _end = input_size // num_mems * (i + 1)
            mem_ins.append(self.memory_in[:, _start:_end])
        self.value_memory_in = mem_ins[0]
        self.q1_memory_in = mem_ins[1]
        self.q2_memory_in = mem_ins[2]

    def _create_observation_in(self, vis_encode_type):
        """
        Creates the observation inputs, and a CNN if needed,
        :param vis_encode_type: Type of CNN encoder.
        :param share_ac_cnn: Whether or not to share the actor and critic CNNs.
        :return A tuple of (hidden_policy, hidden_critic). We don't save it to self since they're used
        once and thrown away.
        """
        with tf.variable_scope(POLICY_SCOPE):
            hidden_streams = ModelUtils.create_observation_streams(
                self.policy.visual_in,
                self.policy.processed_vector_in,
                1,
                self.h_size,
                0,
                vis_encode_type=vis_encode_type,
                stream_scopes=["critic/value/"],
            )
        hidden_critic = hidden_streams[0]
        return hidden_critic

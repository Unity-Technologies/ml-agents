from enum import Enum
from typing import Callable, Dict, List, Tuple, NamedTuple

import numpy as np
from mlagents.tf_utils import tf

from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.brain import CameraResolution

ActivationFunction = Callable[[tf.Tensor], tf.Tensor]
EncoderFunction = Callable[
    [tf.Tensor, int, ActivationFunction, int, str, bool], tf.Tensor
]

EPSILON = 1e-7


class EncoderType(Enum):
    SIMPLE = "simple"
    NATURE_CNN = "nature_cnn"
    RESNET = "resnet"


class LearningRateSchedule(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


class NormalizerTensors(NamedTuple):
    update_op: tf.Operation
    steps: tf.Tensor
    running_mean: tf.Tensor
    running_variance: tf.Tensor


class ModelUtils:
    # Minimum supported side for each encoder type. If refactoring an encoder, please
    # adjust these also.
    MIN_RESOLUTION_FOR_ENCODER = {
        EncoderType.SIMPLE: 20,
        EncoderType.NATURE_CNN: 36,
        EncoderType.RESNET: 15,
    }

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(
            0, name="global_step", trainable=False, dtype=tf.int32
        )
        steps_to_increment = tf.placeholder(
            shape=[], dtype=tf.int32, name="steps_to_increment"
        )
        increment_step = tf.assign(global_step, tf.add(global_step, steps_to_increment))
        return global_step, increment_step, steps_to_increment

    @staticmethod
    def create_learning_rate(
        lr_schedule: LearningRateSchedule,
        lr: float,
        global_step: tf.Tensor,
        max_step: int,
    ) -> tf.Tensor:
        """
        Create a learning rate tensor.
        :param lr_schedule: Type of learning rate schedule.
        :param lr: Base learning rate.
        :param global_step: A TF Tensor representing the total global step.
        :param max_step: The maximum number of steps in the training run.
        :return: A Tensor containing the learning rate.
        """
        if lr_schedule == LearningRateSchedule.CONSTANT:
            learning_rate = tf.Variable(lr)
        elif lr_schedule == LearningRateSchedule.LINEAR:
            learning_rate = tf.train.polynomial_decay(
                lr, global_step, max_step, 1e-10, power=1.0
            )
        else:
            raise UnityTrainerException(
                "The learning rate schedule {} is invalid.".format(lr_schedule)
            )
        return learning_rate

    @staticmethod
    def scaled_init(scale):
        return tf.initializers.variance_scaling(scale)

    @staticmethod
    def swish(input_activation: tf.Tensor) -> tf.Tensor:
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))

    @staticmethod
    def create_visual_input(
        camera_parameters: CameraResolution, name: str
    ) -> tf.Tensor:
        """
        Creates image input op.
        :param camera_parameters: Parameters for visual observation.
        :param name: Desired name of input op.
        :return: input op.
        """
        o_size_h = camera_parameters.height
        o_size_w = camera_parameters.width
        c_channels = camera_parameters.num_channels

        visual_in = tf.placeholder(
            shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32, name=name
        )
        return visual_in

    @staticmethod
    def create_visual_input_placeholders(
        camera_resolutions: List[CameraResolution]
    ) -> List[tf.Tensor]:
        """
        Creates input placeholders for visual inputs.
        :param camera_resolutions: A List of CameraResolutions that specify the resolutions
        of the input visual observations.
        :returns: A List of Tensorflow placeholders where the input iamges should be fed.
        """
        visual_in: List[tf.Tensor] = []
        for i, camera_resolution in enumerate(camera_resolutions):
            visual_input = ModelUtils.create_visual_input(
                camera_resolution, name="visual_observation_" + str(i)
            )
            visual_in.append(visual_input)
        return visual_in

    @staticmethod
    def create_vector_input(
        vec_obs_size: int, name: str = "vector_observation"
    ) -> tf.Tensor:
        """
        Creates ops for vector observation input.
        :param vec_obs_size: Size of stacked vector observation.
        :param name: Name of the placeholder op.
        :return: Placeholder for vector observations.
        """
        vector_in = tf.placeholder(
            shape=[None, vec_obs_size], dtype=tf.float32, name=name
        )
        return vector_in

    @staticmethod
    def normalize_vector_obs(
        vector_obs: tf.Tensor,
        running_mean: tf.Tensor,
        running_variance: tf.Tensor,
        normalization_steps: tf.Tensor,
    ) -> tf.Tensor:
        """
        Create a normalized version of an input tensor.
        :param vector_obs: Input vector observation tensor.
        :param running_mean: Tensorflow tensor representing the current running mean.
        :param running_variance: Tensorflow tensor representing the current running variance.
        :param normalization_steps: Tensorflow tensor representing the current number of normalization_steps.
        :return: A normalized version of vector_obs.
        """
        normalized_state = tf.clip_by_value(
            (vector_obs - running_mean)
            / tf.sqrt(
                running_variance / (tf.cast(normalization_steps, tf.float32) + 1)
            ),
            -5,
            5,
            name="normalized_state",
        )
        return normalized_state

    @staticmethod
    def create_normalizer(vector_obs: tf.Tensor) -> NormalizerTensors:
        """
        Creates the normalizer and the variables required to store its state.
        :param vector_obs: A Tensor representing the next value to normalize. When the
            update operation is called, it will use vector_obs to update the running mean
            and variance.
        :return: A NormalizerTensors tuple that holds running mean, running variance, number of steps,
            and the update operation.
        """

        vec_obs_size = vector_obs.shape[1]
        steps = tf.get_variable(
            "normalization_steps",
            [],
            trainable=False,
            dtype=tf.int32,
            initializer=tf.zeros_initializer(),
        )
        running_mean = tf.get_variable(
            "running_mean",
            [vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
        )
        running_variance = tf.get_variable(
            "running_variance",
            [vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initializer=tf.ones_initializer(),
        )
        update_normalization = ModelUtils.create_normalizer_update(
            vector_obs, steps, running_mean, running_variance
        )
        return NormalizerTensors(
            update_normalization, steps, running_mean, running_variance
        )

    @staticmethod
    def create_normalizer_update(
        vector_input: tf.Tensor,
        steps: tf.Tensor,
        running_mean: tf.Tensor,
        running_variance: tf.Tensor,
    ) -> tf.Operation:
        """
        Creates the update operation for the normalizer.
        :param vector_input: Vector observation to use for updating the running mean and variance.
        :param running_mean: Tensorflow tensor representing the current running mean.
        :param running_variance: Tensorflow tensor representing the current running variance.
        :param steps: Tensorflow tensor representing the current number of steps that have been normalized.
        :return: A TF operation that updates the normalization based on vector_input.
        """
        # Based on Welford's algorithm for running mean and standard deviation, for batch updates. Discussion here:
        # https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
        steps_increment = tf.shape(vector_input)[0]
        total_new_steps = tf.add(steps, steps_increment)

        # Compute the incremental update and divide by the number of new steps.
        input_to_old_mean = tf.subtract(vector_input, running_mean)
        new_mean = running_mean + tf.reduce_sum(
            input_to_old_mean / tf.cast(total_new_steps, dtype=tf.float32), axis=0
        )
        # Compute difference of input to the new mean for Welford update
        input_to_new_mean = tf.subtract(vector_input, new_mean)
        new_variance = running_variance + tf.reduce_sum(
            input_to_new_mean * input_to_old_mean, axis=0
        )
        update_mean = tf.assign(running_mean, new_mean)
        update_variance = tf.assign(running_variance, new_variance)
        update_norm_step = tf.assign(steps, total_new_steps)
        return tf.group([update_mean, update_variance, update_norm_step])

    @staticmethod
    def create_vector_observation_encoder(
        observation_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
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
                    kernel_initializer=tf.initializers.variance_scaling(1.0),
                )
        return hidden

    @staticmethod
    def create_visual_observation_encoder(
        image_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
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
            hidden = tf.layers.flatten(conv2)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = ModelUtils.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def create_nature_cnn_visual_observation_encoder(
        image_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            conv1 = tf.layers.conv2d(
                image_input,
                32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_1",
            )
            conv2 = tf.layers.conv2d(
                conv1,
                64,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_2",
            )
            conv3 = tf.layers.conv2d(
                conv2,
                64,
                kernel_size=[3, 3],
                strides=[1, 1],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_3",
            )
            hidden = tf.layers.flatten(conv3)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = ModelUtils.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def create_resnet_visual_observation_encoder(
        image_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        with tf.variable_scope(scope):
            hidden = image_input
            for i, ch in enumerate(n_channels):
                hidden = tf.layers.conv2d(
                    hidden,
                    ch,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    reuse=reuse,
                    name="layer%dconv_1" % i,
                )
                hidden = tf.layers.max_pooling2d(
                    hidden, pool_size=[3, 3], strides=[2, 2], padding="same"
                )
                # create residual blocks
                for j in range(n_blocks):
                    block_input = hidden
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=reuse,
                        name="layer%d_%d_conv1" % (i, j),
                    )
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=reuse,
                        name="layer%d_%d_conv2" % (i, j),
                    )
                    hidden = tf.add(block_input, hidden)
            hidden = tf.nn.relu(hidden)
            hidden = tf.layers.flatten(hidden)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = ModelUtils.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def get_encoder_for_type(encoder_type: EncoderType) -> EncoderFunction:
        ENCODER_FUNCTION_BY_TYPE = {
            EncoderType.SIMPLE: ModelUtils.create_visual_observation_encoder,
            EncoderType.NATURE_CNN: ModelUtils.create_nature_cnn_visual_observation_encoder,
            EncoderType.RESNET: ModelUtils.create_resnet_visual_observation_encoder,
        }
        return ENCODER_FUNCTION_BY_TYPE.get(
            encoder_type, ModelUtils.create_visual_observation_encoder
        )

    @staticmethod
    def break_into_branches(
        concatenated_logits: tf.Tensor, action_size: List[int]
    ) -> List[tf.Tensor]:
        """
        Takes a concatenated set of logits that represent multiple discrete action branches
        and breaks it up into one Tensor per branch.
        :param concatenated_logits: Tensor that represents the concatenated action branches
        :param action_size: List of ints containing the number of possible actions for each branch.
        :return: A List of Tensors containing one tensor per branch.
        """
        action_idx = [0] + list(np.cumsum(action_size))
        branched_logits = [
            concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        return branched_logits

    @staticmethod
    def create_discrete_action_masking_layer(
        branches_logits: List[tf.Tensor],
        action_masks: tf.Tensor,
        action_size: List[int],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Creates a masking layer for the discrete actions
        :param branches_logits: A List of the unnormalized action probabilities for each branch
        :param action_masks: The mask for the logits. Must be of dimension [None x total_number_of_action]
        :param action_size: A list containing the number of possible actions for each branch
        :return: The action output dimension [batch_size, num_branches], the concatenated
            normalized probs (after softmax)
        and the concatenated normalized log probs
        """
        branch_masks = ModelUtils.break_into_branches(action_masks, action_size)
        raw_probs = [
            tf.multiply(tf.nn.softmax(branches_logits[k]) + EPSILON, branch_masks[k])
            for k in range(len(action_size))
        ]
        normalized_probs = [
            tf.divide(raw_probs[k], tf.reduce_sum(raw_probs[k], axis=1, keepdims=True))
            for k in range(len(action_size))
        ]
        output = tf.concat(
            [
                tf.multinomial(tf.log(normalized_probs[k] + EPSILON), 1)
                for k in range(len(action_size))
            ],
            axis=1,
        )
        return (
            output,
            tf.concat([normalized_probs[k] for k in range(len(action_size))], axis=1),
            tf.concat(
                [
                    tf.log(normalized_probs[k] + EPSILON)
                    for k in range(len(action_size))
                ],
                axis=1,
            ),
        )

    @staticmethod
    def _check_resolution_for_encoder(
        vis_in: tf.Tensor, vis_encoder_type: EncoderType
    ) -> None:
        min_res = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[vis_encoder_type]
        height = vis_in.shape[1]
        width = vis_in.shape[2]
        if height < min_res or width < min_res:
            raise UnityTrainerException(
                f"Visual observation resolution ({width}x{height}) is too small for"
                f"the provided EncoderType ({vis_encoder_type.value}). The min dimension is {min_res}"
            )

    @staticmethod
    def create_observation_streams(
        visual_in: List[tf.Tensor],
        vector_in: tf.Tensor,
        num_streams: int,
        h_size: int,
        num_layers: int,
        vis_encode_type: EncoderType = EncoderType.SIMPLE,
        stream_scopes: List[str] = None,
    ) -> List[tf.Tensor]:
        """
        Creates encoding stream for observations.
        :param num_streams: Number of streams to create.
        :param h_size: Size of hidden linear layers in stream.
        :param num_layers: Number of hidden linear layers in stream.
        :param stream_scopes: List of strings (length == num_streams), which contains
            the scopes for each of the streams. None if all under the same TF scope.
        :return: List of encoded streams.
        """
        activation_fn = ModelUtils.swish
        vector_observation_input = vector_in

        final_hiddens = []
        for i in range(num_streams):
            # Pick the encoder function based on the EncoderType
            create_encoder_func = ModelUtils.get_encoder_for_type(vis_encode_type)

            visual_encoders = []
            hidden_state, hidden_visual = None, None
            _scope_add = stream_scopes[i] if stream_scopes else ""
            if len(visual_in) > 0:
                for j, vis_in in enumerate(visual_in):
                    ModelUtils._check_resolution_for_encoder(vis_in, vis_encode_type)
                    encoded_visual = create_encoder_func(
                        vis_in,
                        h_size,
                        activation_fn,
                        num_layers,
                        f"{_scope_add}main_graph_{i}_encoder{j}",  # scope
                        False,  # reuse
                    )
                    visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if vector_in.get_shape()[-1] > 0:  # Don't encode 0-shape inputs
                hidden_state = ModelUtils.create_vector_observation_encoder(
                    vector_observation_input,
                    h_size,
                    activation_fn,
                    num_layers,
                    scope=f"{_scope_add}main_graph_{i}",
                    reuse=False,
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
        half_point = int(m_size / 2)
        with tf.variable_scope(name):
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(half_point)
            lstm_vector_in = tf.nn.rnn_cell.LSTMStateTuple(
                memory_in[:, :half_point], memory_in[:, half_point:]
            )
            recurrent_output, lstm_state_out = tf.nn.dynamic_rnn(
                rnn_cell, lstm_input_state, initial_state=lstm_vector_in
            )

        recurrent_output = tf.reshape(recurrent_output, shape=[-1, half_point])
        return recurrent_output, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

    @staticmethod
    def create_value_heads(
        stream_names: List[str], hidden_input: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        value_heads = {}
        for name in stream_names:
            value = tf.layers.dense(hidden_input, 1, name="{}_value".format(name))
            value_heads[name] = value
        value = tf.reduce_mean(list(value_heads.values()), 0)
        return value_heads, value

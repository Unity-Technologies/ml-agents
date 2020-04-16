from enum import Enum
from typing import Callable, List, NamedTuple

import numpy as np
import torch
from torch import nn

from mlagents.trainers.exception import UnityTrainerException

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool], torch.Tensor
]

EPSILON = 1e-7


class EncoderType(Enum):
    SIMPLE = "simple"
    NATURE_CNN = "nature_cnn"
    RESNET = "resnet"


class ActionType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class LearningRateSchedule(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


class NormalizerTensors(NamedTuple):
    steps: torch.Tensor
    running_mean: torch.Tensor
    running_variance: torch.Tensor


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        print(vec_obs_size)
        self.normalization_steps = torch.tensor(1)
        self.running_mean = torch.zeros(vec_obs_size)
        self.running_variance = torch.ones(vec_obs_size)

    def forward(self, inputs):
        inputs = torch.from_numpy(inputs)
        normalized_state = torch.clamp(
            (inputs - self.running_mean)
            / torch.sqrt(
                self.running_variance / self.normalization_steps.type(torch.float32)
            ),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input):
        vector_input = torch.from_numpy(vector_input)
        mean_current_observation = vector_input.mean(0).type(torch.float32)
        new_mean = self.running_mean + (
            mean_current_observation - self.running_mean
        ) / (self.normalization_steps + 1).type(torch.float32)
        new_variance = self.running_variance + (mean_current_observation - new_mean) * (
            mean_current_observation - self.running_mean
        )
        self.running_mean = new_mean
        self.running_variance = new_variance
        self.normalization_steps = self.normalization_steps + 1


class ValueHeads(nn.Module):
    def __init__(self, stream_names, input_size):
        super(ValueHeads, self).__init__()
        self.stream_names = stream_names
        self.value_heads = {}

        for name in stream_names:
            value = nn.Linear(input_size, 1)
            self.value_heads[name] = value

    def forward(self, hidden):
        value_outputs = {}
        for stream_name, _ in self.value_heads.items():
            value_outputs[stream_name] = self.value_heads[stream_name](hidden)
        return value_outputs, torch.mean(torch.stack(list(value_outputs)), dim=0)


class VectorEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(VectorEncoder, self).__init__(**kwargs)
        self.layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        print(self.layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleVisualEncoder(nn.Module):
    def __init__(self, initial_channels):
        super(SimpleVisualEncoder, self).__init__()
        self.conv1 = nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(16, 32, [4, 4], [2, 2])

    def forward(self, visual_obs):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        return torch.flatten(conv_2)


class NatureVisualEncoder(nn.Module):
    def __init__(self, initial_channels):
        super(NatureVisualEncoder, self).__init__()
        self.conv1 = nn.Conv2d(initial_channels, 32, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(43, 64, [4, 4], [2, 2])
        self.conv3 = nn.Conv2d(64, 64, [3, 3], [1, 1])

    def forward(self, visual_obs):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        conv_3 = torch.relu(self.conv3(conv_2))
        return torch.flatten(conv_3)


class DiscreteActionMask(nn.Module):
    def __init__(self, action_size):
        super(DiscreteActionMask, self).__init__()
        self.action_size = action_size

    @staticmethod
    def break_into_branches(
        concatenated_logits: torch.Tensor, action_size: List[int]
    ) -> List[torch.Tensor]:
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

    def forward(self, branches_logits, action_masks):
        branch_masks = self.break_into_branches(action_masks, self.action_size)
        raw_probs = [
            torch.multiply(
                torch.softmax(branches_logits[k], dim=-1) + EPSILON, branch_masks[k]
            )
            for k in range(len(self.action_size))
        ]
        normalized_probs = [
            torch.divide(raw_probs[k], torch.sum(raw_probs[k], dim=1, keepdims=True))
            for k in range(len(self.action_size))
        ]
        output = torch.cat(
            [
                torch.multinomial(torch.log(normalized_probs[k] + EPSILON), 1)
                for k in range(len(self.action_size))
            ],
            dim=1,
        )
        return (
            output,
            torch.cat(
                [normalized_probs[k] for k in range(len(self.action_size))], dim=1
            ),
            torch.cat(
                [
                    torch.log(normalized_probs[k] + EPSILON)
                    for k in range(len(self.action_size))
                ],
                axis=1,
            ),
        )


class GlobalSteps(nn.Module):
    def __init__(self):
        super(GlobalSteps, self).__init__()
        self.global_step = torch.Tensor([0])

    def increment(self, value):
        self.global_step += value


class LearningRate(nn.Module):
    def __init__(self, lr):
        # Todo: add learning rate decay
        super(LearningRate, self).__init__()
        self.learning_rate = torch.Tensor([lr])


class ResNetVisualEncoder(nn.Module):
    def __init__(self, initial_channels):
        super(ResNetVisualEncoder, self).__init__()
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        self.layers = []
        for _, channel in enumerate(n_channels):
            self.layers.append(nn.Conv2d(initial_channels, channel, [3, 3], [1, 1]))
            self.layers.append(nn.MaxPool2d([3, 3], [2, 2]))
            for _ in range(n_blocks):
                self.layers.append(self.make_block(channel))
        self.layers.append(nn.RELU())

    @staticmethod
    def make_block(channel):
        block_layers = [
            nn.ReLU(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1]),
            nn.ReLU(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1]),
        ]
        return block_layers

    @staticmethod
    def forward_block(input_hidden, block_layers):
        hidden = input_hidden
        for layer in block_layers:
            hidden = layer(hidden)
        return hidden + input_hidden

    def forward(self, visual_obs):
        hidden = visual_obs
        for layer in self.layers:
            if layer is nn.Module:
                hidden = layer(hidden)
            elif layer is list:
                hidden = self.forward_block(hidden, layer)
        return hidden.flatten()


class ModelUtils:
    # Minimum supported side for each encoder type. If refactoring an encoder, please
    # adjust these also.
    MIN_RESOLUTION_FOR_ENCODER = {
        EncoderType.SIMPLE: 20,
        EncoderType.NATURE_CNN: 36,
        EncoderType.RESNET: 15,
    }

    # @staticmethod
    # def scaled_init(scale):
    #    return tf.initializers.variance_scaling(scale)

    @staticmethod
    def swish(input_activation: torch.Tensor) -> torch.Tensor:
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return torch.mul(input_activation, torch.sigmoid(input_activation))

    @staticmethod
    def get_encoder_for_type(encoder_type: EncoderType) -> nn.Module:
        ENCODER_FUNCTION_BY_TYPE = {
            EncoderType.SIMPLE: SimpleVisualEncoder,
            EncoderType.NATURE_CNN: NatureVisualEncoder,
            EncoderType.RESNET: ResNetVisualEncoder,
        }
        return ENCODER_FUNCTION_BY_TYPE.get(encoder_type)

    @staticmethod
    def _check_resolution_for_encoder(
        vis_in: torch.Tensor, vis_encoder_type: EncoderType
    ) -> None:
        min_res = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[vis_encoder_type]
        height = vis_in.shape[1]
        width = vis_in.shape[2]
        if height < min_res or width < min_res:
            raise UnityTrainerException(
                f"Visual observation resolution ({width}x{height}) is too small for"
                f"the provided EncoderType ({vis_encoder_type.value}). The min dimension is {min_res}"
            )

    # @staticmethod
    # def compose_streams(
    #     visual_in: List[torch.Tensor],
    #     vector_in: torch.Tensor,
    #     num_streams: int,
    #     h_size: int,
    #     num_layers: int,
    #     vis_encode_type: EncoderType = EncoderType.SIMPLE,
    #     stream_scopes: List[str] = None,
    # ) -> List[torch.Tensor]:
    #     """
    #     Creates encoding stream for observations.
    #     :param num_streams: Number of streams to create.
    #     :param h_size: Size of hidden linear layers in stream.
    #     :param num_layers: Number of hidden linear layers in stream.
    #     :param stream_scopes: List of strings (length == num_streams), which contains
    #         the scopes for each of the streams. None if all under the same TF scope.
    #     :return: List of encoded streams.
    #     """
    #     activation_fn = ModelUtils.swish
    #     vector_observation_input = vector_in

    #     final_hiddens = []
    #     for i in range(num_streams):
    #         # Pick the encoder function based on the EncoderType
    #         create_encoder_func = ModelUtils.get_encoder_for_type(vis_encode_type)

    #         visual_encoders = []
    #         hidden_state, hidden_visual = None, None
    #         _scope_add = stream_scopes[i] if stream_scopes else ""
    #         if len(visual_in) > 0:
    #             for j, vis_in in enumerate(visual_in):
    #                 ModelUtils._check_resolution_for_encoder(vis_in, vis_encode_type)
    #                 encoded_visual = create_encoder_func(
    #                     vis_in,
    #                     h_size,
    #                     activation_fn,
    #                     num_layers,
    #                     f"{_scope_add}main_graph_{i}_encoder{j}",  # scope
    #                     False,  # reuse
    #                 )
    #                 visual_encoders.append(encoded_visual)
    #             hidden_visual = torch.cat(visual_encoders, axis=1)
    #         if vector_in.get_shape()[-1] > 0:  # Don't encode 0-shape inputs
    #             hidden_state = ModelUtils.create_vector_observation_encoder(
    #                 vector_observation_input,
    #                 h_size,
    #                 activation_fn,
    #                 num_layers,
    #                 scope=f"{_scope_add}main_graph_{i}",
    #                 reuse=False,
    #             )
    #         if hidden_state is not None and hidden_visual is not None:
    #             final_hidden = torch.cat([hidden_visual, hidden_state], axis=1)
    #         elif hidden_state is None and hidden_visual is not None:
    #             final_hidden = hidden_visual
    #         elif hidden_state is not None and hidden_visual is None:
    #             final_hidden = hidden_state
    #         else:
    #             raise Exception(
    #                 "No valid network configuration possible. "
    #                 "There are no states or observations in this brain"
    #             )
    #         final_hiddens.append(final_hidden)
    #     return final_hiddens

    # @staticmethod
    # def create_recurrent_encoder(input_state, memory_in, sequence_length, name="lstm"):
    #     """
    #     Builds a recurrent encoder for either state or observations (LSTM).
    #     :param sequence_length: Length of sequence to unroll.
    #     :param input_state: The input tensor to the LSTM cell.
    #     :param memory_in: The input memory to the LSTM cell.
    #     :param name: The scope of the LSTM cell.
    #     """
    #     s_size = input_state.get_shape().as_list()[1]
    #     m_size = memory_in.get_shape().as_list()[1]
    #     lstm_input_state = tf.reshape(input_state, shape=[-1, sequence_length, s_size])
    #     memory_in = tf.reshape(memory_in[:, :], [-1, m_size])
    #     half_point = int(m_size / 2)
    #     with tf.variable_scope(name):
    #         rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(half_point)
    #         lstm_vector_in = tf.nn.rnn_cell.LSTMStateTuple(
    #             memory_in[:, :half_point], memory_in[:, half_point:]
    #         )
    #         recurrent_output, lstm_state_out = tf.nn.dynamic_rnn(
    #             rnn_cell, lstm_input_state, initial_state=lstm_vector_in
    #         )

    #     recurrent_output = tf.reshape(recurrent_output, shape=[-1, half_point])
    #     return recurrent_output, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

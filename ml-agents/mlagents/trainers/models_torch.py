from enum import Enum
from typing import Callable, List, NamedTuple

import numpy as np
import torch
from torch import nn

from mlagents.trainers.distributions_torch import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)
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


class NetworkBody(nn.Module):
    def __init__(
        self,
        vector_sizes,
        visual_sizes,
        h_size,
        normalize,
        num_layers,
        m_size,
        vis_encode_type,
        use_lstm,
    ):
        super(NetworkBody, self).__init__()
        self.normalize = normalize
        self.visual_encoders = []
        self.vector_encoders = []
        self.vector_normalizers = []
        self.use_lstm = use_lstm
        self.h_size = h_size
        self.m_size = m_size

        visual_encoder = ModelUtils.get_encoder_for_type(vis_encode_type)
        for vector_size in vector_sizes:
            self.vector_normalizers.append(Normalizer(vector_size))
            self.vector_encoders.append(VectorEncoder(vector_size, h_size, num_layers))
        for visual_size in visual_sizes:
            self.visual_encoders.append(visual_encoder(visual_size))

        if use_lstm:
            self.lstm = nn.LSTM(h_size, h_size, 1)

    def clear_memory(self, batch_size):
        self.memory = (
            torch.zeros(1, batch_size, self.m_size),
            torch.zeros(1, batch_size, self.m_size),
        )

    def update_normalization(self, inputs):
        if self.normalize:
            self.normalizer.update(inputs)

    def forward(self, vec_inputs, vis_inputs):
        vec_embeds = []
        for idx, encoder in enumerate(self.vector_encoders):
            vec_input = vec_inputs[idx]
            if self.normalize:
                vec_input = self.normalizers[idx](vec_inputs[idx])
            hidden = encoder(vec_input)
            vec_embeds.append(hidden)

        vis_embeds = []
        for idx, encoder in enumerate(self.visual_encoders):
            hidden = encoder(vis_inputs[idx])
            vis_embeds.append(hidden)

        vec_embeds = torch.cat(vec_embeds)
        vis_embeds = torch.cat(vis_embeds)
        embedding = torch.cat([vec_embeds, vis_embeds])
        if self.use_lstm:
            embedding, self.memory = self.lstm(embedding, self.memory)
        return embedding


class Actor(nn.Module):
    def __init__(
        self,
        h_size,
        vector_sizes,
        visual_sizes,
        act_size,
        normalize,
        num_layers,
        m_size,
        vis_encode_type,
        act_type,
        use_lstm,
    ):
        super(Actor, self).__init__()
        self.act_type = act_type
        self.act_size = act_size
        self.network_body = NetworkBody(
            vector_sizes,
            visual_sizes,
            h_size,
            normalize,
            num_layers,
            m_size,
            vis_encode_type,
            use_lstm,
        )
        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(h_size, act_size)
        else:
            self.distribution = MultiCategoricalDistribution(h_size, act_size)

    def forward(self, vec_inputs, vis_inputs, masks=None):
        embedding = self.network_body(vec_inputs, vis_inputs)
        if self.act_type == ActionType.CONTINUOUS:
            dist = self.distribution(embedding)
        else:
            dist = self.distribution(embedding, masks=masks)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        stream_names,
        h_size,
        vector_sizes,
        visual_sizes,
        normalize,
        num_layers,
        m_size,
        vis_encode_type,
        use_lstm,
    ):
        super(Critic, self).__init__()
        self.stream_names = stream_names
        self.network_body = NetworkBody(
            vector_sizes,
            visual_sizes,
            h_size,
            normalize,
            num_layers,
            m_size,
            vis_encode_type,
            use_lstm,
        )
        self.value_heads = ValueHeads(stream_names, h_size)

    def forward(self, vec_inputs, vis_inputs):
        embedding = self.network_body(vec_inputs, vis_inputs)
        return self.value_heads(embedding)


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
            torch.mul(
                torch.softmax(branches_logits[k], dim=-1) + EPSILON, branch_masks[k]
            )
            for k in range(len(self.action_size))
        ]
        normalized_probs = [
            torch.div(raw_probs[k], torch.sum(raw_probs[k], dim=1, keepdims=True))
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
        self.layers.append(nn.ReLU())

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

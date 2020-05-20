from enum import Enum
from typing import Callable, NamedTuple

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

    @staticmethod
    def from_str(label):
        if label in "continuous":
            return ActionType.CONTINUOUS
        elif label in "discrete":
            return ActionType.DISCRETE
        else:
            raise NotImplementedError


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
            if vector_size != 0:
                self.vector_normalizers.append(Normalizer(vector_size))
                self.vector_encoders.append(
                    VectorEncoder(vector_size, h_size, num_layers)
                )
        for visual_size in visual_sizes:
            self.visual_encoders.append(
                visual_encoder(
                    visual_size.height,
                    visual_size.width,
                    visual_size.num_channels,
                    h_size,
                )
            )

        self.vector_encoders = nn.ModuleList(self.vector_encoders)
        self.visual_encoders = nn.ModuleList(self.visual_encoders)
        if use_lstm:
            self.lstm = nn.LSTM(h_size, m_size // 2, 1)

    def update_normalization(self, vec_inputs):
        if self.normalize:
            for idx, vec_input in enumerate(vec_inputs):
                self.vector_normalizers[idx].update(vec_input)

    def forward(self, vec_inputs, vis_inputs, memories=None, sequence_length=1):
        vec_embeds = []
        for idx, encoder in enumerate(self.vector_encoders):
            vec_input = vec_inputs[idx]
            if self.normalize:
                vec_input = self.vector_normalizers[idx](vec_input)
            hidden = encoder(vec_input)
            vec_embeds.append(hidden)

        vis_embeds = []
        for idx, encoder in enumerate(self.visual_encoders):
            vis_input = vis_inputs[idx]
            vis_input = vis_input.permute([0, 3, 1, 2])
            hidden = encoder(vis_input)
            vis_embeds.append(hidden)

        embedding = torch.cat(vec_embeds + vis_embeds)

        if self.use_lstm:
            embedding = embedding.reshape([sequence_length, -1, self.h_size])
            memories = torch.split(memories, self.m_size // 2, dim=-1)
            embedding, memories = self.lstm(embedding, memories)
            embedding = embedding.reshape([-1, self.m_size // 2])
            memories = torch.cat(memories, dim=-1)
        return embedding, memories


class ActorCritic(nn.Module):
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
        stream_names,
        separate_critic,
    ):
        super(ActorCritic, self).__init__()
        self.act_type = ActionType.from_str(act_type)
        self.act_size = act_size
        self.separate_critic = separate_critic
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
        if use_lstm:
            embedding_size = m_size // 2
        else:
            embedding_size = h_size
        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(embedding_size, act_size[0])
        else:
            self.distribution = MultiCategoricalDistribution(embedding_size, act_size)
        if separate_critic:
            self.critic = Critic(
                stream_names,
                h_size,
                vector_sizes,
                visual_sizes,
                normalize,
                num_layers,
                m_size,
                vis_encode_type,
            )
        else:
            self.stream_names = stream_names
            self.value_heads = ValueHeads(stream_names, embedding_size)

    def update_normalization(self, vector_obs):
        self.network_body.update_normalization(vector_obs)
        if self.separate_critic:
            self.critic.network_body.update_normalization(vector_obs)

    def critic_pass(self, vec_inputs, vis_inputs, memories=None):
        if self.separate_critic:
            return self.critic(vec_inputs, vis_inputs)
        else:
            embedding, _ = self.network_body(vec_inputs, vis_inputs, memories=memories)
            return self.value_heads(embedding)

    def sample_action(self, dists):
        actions = []
        for action_dist in dists:
            action = action_dist.sample()
            actions.append(action)
        actions = torch.stack(actions, dim=-1)
        return actions

    def get_probs_and_entropy(self, actions, dists):
        log_probs = []
        entropies = []
        for idx, action_dist in enumerate(dists):
            action = actions[..., idx]
            log_probs.append(action_dist.log_prob(action))
            entropies.append(action_dist.entropy())
        log_probs = torch.stack(log_probs, dim=-1)
        entropies = torch.stack(entropies, dim=-1)
        if self.act_type == ActionType.CONTINUOUS:
            log_probs = log_probs.squeeze(-1)
            entropies = entropies.squeeze(-1)
        return log_probs, entropies

    def evaluate(
        self, vec_inputs, vis_inputs, masks=None, memories=None, sequence_length=1
    ):
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, memories, sequence_length
        )
        dists = self.distribution(embedding, masks=masks)

        return dists, memories

    def forward(
        self, vec_inputs, vis_inputs, masks=None, memories=None, sequence_length=1
    ):
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, memories, sequence_length
        )
        value_outputs = self.critic(vec_inputs, vis_inputs)
        dists = self.distribution(embedding, masks=masks)
        return dists, value_outputs, memories


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
    ):
        super(Critic, self).__init__()
        self.network_body = NetworkBody(
            vector_sizes,
            visual_sizes,
            h_size,
            normalize,
            num_layers,
            m_size,
            vis_encode_type,
            False,
        )
        self.stream_names = stream_names
        self.value_heads = ValueHeads(stream_names, h_size)

    def forward(self, vec_inputs, vis_inputs):
        embedding, _ = self.network_body(vec_inputs, vis_inputs)
        return self.value_heads(embedding)


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.normalization_steps = torch.tensor(1)
        self.running_mean = torch.zeros(vec_obs_size)
        self.running_variance = torch.ones(vec_obs_size)

    def forward(self, inputs):
        normalized_state = torch.clamp(
            (inputs - self.running_mean)
            / torch.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input):
        steps_increment = vector_input.size()[0]
        total_new_steps = self.normalization_steps + steps_increment

        input_to_old_mean = vector_input - self.running_mean
        new_mean = self.running_mean + (input_to_old_mean / total_new_steps).sum(0)

        input_to_new_mean = vector_input - new_mean
        new_variance = self.running_variance + (
            input_to_new_mean * input_to_old_mean
        ).sum(0)
        self.running_mean = new_mean
        self.running_variance = new_variance
        self.normalization_steps = total_new_steps


class ValueHeads(nn.Module):
    def __init__(self, stream_names, input_size):
        super(ValueHeads, self).__init__()
        self.stream_names = stream_names
        self.value_heads = {}

        for name in stream_names:
            value = nn.Linear(input_size, 1)
            self.value_heads[name] = value
        self.value_heads = nn.ModuleDict(self.value_heads)

    def forward(self, hidden):
        value_outputs = {}
        for stream_name, _ in self.value_heads.items():
            value_outputs[stream_name] = self.value_heads[stream_name](hidden).squeeze(
                -1
            )
        return (
            value_outputs,
            torch.mean(torch.stack(list(value_outputs.values())), dim=0),
        )


class VectorEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(VectorEncoder, self).__init__(**kwargs)
        self.layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


class SimpleVisualEncoder(nn.Module):
    def __init__(self, height, width, initial_channels, output_size):
        super(SimpleVisualEncoder, self).__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

        self.conv1 = nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(16, 32, [4, 4], [2, 2])
        self.dense = nn.Linear(self.final_flat, self.h_size)

    def forward(self, visual_obs):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        hidden = self.dense(conv_2.reshape([-1, self.final_flat]))
        return hidden


class NatureVisualEncoder(nn.Module):
    def __init__(self, height, width, initial_channels, output_size):
        super(NatureVisualEncoder, self).__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 64

        self.conv1 = nn.Conv2d(initial_channels, 32, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(43, 64, [4, 4], [2, 2])
        self.conv3 = nn.Conv2d(64, 64, [3, 3], [1, 1])
        self.dense = nn.Linear(self.final_flat, self.h_size)

    def forward(self, visual_obs):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        conv_3 = torch.relu(self.conv3(conv_2))
        hidden = self.dense(conv_3.reshape([-1, self.final_flat]))
        return hidden


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

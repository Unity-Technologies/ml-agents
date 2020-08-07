from typing import Tuple, Optional

from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch.layers import linear_layer, Initialization, Swish

import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size: int):
        super().__init__()
        self.register_buffer("normalization_steps", torch.tensor(1))
        self.register_buffer("running_mean", torch.zeros(vec_obs_size))
        self.register_buffer("running_variance", torch.ones(vec_obs_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized_state = torch.clamp(
            (inputs - self.running_mean)
            / torch.sqrt(self.running_variance / self.normalization_steps),
            -5,
            5,
        )
        return normalized_state

    def update(self, vector_input: torch.Tensor) -> None:
        steps_increment = vector_input.size()[0]
        total_new_steps = self.normalization_steps + steps_increment

        input_to_old_mean = vector_input - self.running_mean
        new_mean = self.running_mean + (input_to_old_mean / total_new_steps).sum(0)

        input_to_new_mean = vector_input - new_mean
        new_variance = self.running_variance + (
            input_to_new_mean * input_to_old_mean
        ).sum(0)
        # Update in-place
        self.running_mean.data.copy_(new_mean.data)
        self.running_variance.data.copy_(new_variance.data)
        self.normalization_steps.data.copy_(total_new_steps.data)

    def copy_from(self, other_normalizer: "Normalizer") -> None:
        self.normalization_steps.data.copy_(other_normalizer.normalization_steps.data)
        self.running_mean.data.copy_(other_normalizer.running_mean.data)
        self.running_variance.copy_(other_normalizer.running_variance.data)


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


def pool_out_shape(h_w: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


class VectorEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        normalize: bool = False,
    ):
        self.normalizer: Optional[Normalizer] = None
        super().__init__()
        self.layers = [
            linear_layer(
                input_size,
                hidden_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.0,
            )
        ]
        if normalize:
            self.normalizer = Normalizer(input_size)

        for _ in range(num_layers - 1):
            self.layers.append(
                linear_layer(
                    hidden_size,
                    hidden_size,
                    kernel_init=Initialization.KaimingHeNormal,
                    kernel_gain=1.0,
                )
            )
            self.layers.append(Swish())
        self.seq_layers = nn.Sequential(*self.layers)

    def forward(self, inputs: torch.Tensor) -> None:
        if self.normalizer is not None:
            inputs = self.normalizer(inputs)
        return self.seq_layers(inputs)

    def copy_normalization(self, other_encoder: "VectorEncoder") -> None:
        if self.normalizer is not None and other_encoder.normalizer is not None:
            self.normalizer.copy_from(other_encoder.normalizer)

    def update_normalization(self, inputs: torch.Tensor) -> None:
        if self.normalizer is not None:
            self.normalizer.update(inputs)


class VectorAndUnnormalizedInputEncoder(VectorEncoder):
    """
    Encoder for concatenated vector input (can be normalized) and unnormalized vector input.
    This is used for passing inputs to the network that should not be normalized, such as
    actions in the case of a Q function or task parameterizations. It will result in an encoder with
    this structure:
    ____________       ____________       ____________
    | Vector     |     | Normalize  |     | Fully      |
    |            | --> |            | --> | Connected  |      ___________
    |____________|     |____________|     |            |     | Output    |
    ____________                          |            | --> |           |
    |Unnormalized|                        |            |     |___________|
    |   Input    | ---------------------> |            |
    |____________|                        |____________|
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        unnormalized_input_size: int,
        num_layers: int,
        normalize: bool = False,
    ):
        super().__init__(
            input_size + unnormalized_input_size,
            hidden_size,
            num_layers,
            normalize=False,
        )
        if normalize:
            self.normalizer = Normalizer(input_size)
        else:
            self.normalizer = None

    def forward(  # pylint: disable=W0221
        self, inputs: torch.Tensor, unnormalized_inputs: Optional[torch.Tensor] = None
    ) -> None:
        if unnormalized_inputs is None:
            raise UnityTrainerException(
                "Attempted to call an VectorAndUnnormalizedInputEncoder without an unnormalized input."
            )  # Fix mypy errors about method parameters.
        if self.normalizer is not None:
            inputs = self.normalizer(inputs)
        return self.seq_layers(torch.cat([inputs, unnormalized_inputs], dim=-1))


class SimpleVisualEncoder(nn.Module):
    def __init__(
        self, height: int, width: int, initial_channels: int, output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32

        self.conv1 = nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(16, 32, [4, 4], [2, 2])
        self.dense = linear_layer(
            self.final_flat,
            self.h_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.0,
        )

    def forward(self, visual_obs: torch.Tensor) -> None:
        conv_1 = nn.functional.leaky_relu(self.conv1(visual_obs))
        conv_2 = nn.functional.leaky_relu(self.conv2(conv_1))
        # hidden = torch.relu(self.dense(conv_2.view([-1, self.final_flat])))
        hidden = nn.functional.leaky_relu(
            self.dense(torch.reshape(conv_2, (-1, self.final_flat)))
        )
        return hidden


class NatureVisualEncoder(nn.Module):
    def __init__(self, height, width, initial_channels, output_size):
        super().__init__()
        self.h_size = output_size
        conv_1_hw = conv_output_shape((height, width), 8, 4)
        conv_2_hw = conv_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 64

        self.conv1 = nn.Conv2d(initial_channels, 32, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(32, 64, [4, 4], [2, 2])
        self.conv3 = nn.Conv2d(64, 64, [3, 3], [1, 1])
        self.dense = linear_layer(
            self.final_flat,
            self.h_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.0,
        )

    def forward(self, visual_obs):
        conv_1 = nn.functional.leaky_relu(self.conv1(visual_obs))
        conv_2 = nn.functional.leaky_relu(self.conv2(conv_1))
        conv_3 = nn.functional.leaky_relu(self.conv3(conv_2))
        hidden = nn.functional.leaky_relu(
            self.dense(conv_3.view([-1, self.final_flat]))
        )
        return hidden


class ResNetVisualEncoder(nn.Module):
    def __init__(self, height, width, initial_channels, final_hidden):
        super().__init__()
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        self.layers = []
        last_channel = initial_channels
        for _, channel in enumerate(n_channels):
            self.layers.append(
                nn.Conv2d(last_channel, channel, [3, 3], [1, 1], padding=1)
            )
            self.layers.append(nn.MaxPool2d([3, 3], [2, 2]))
            height, width = pool_out_shape((height, width), 3)
            for _ in range(n_blocks):
                self.layers.append(self.make_block(channel))
            last_channel = channel
        self.layers.append(Swish())
        self.dense = linear_layer(
            n_channels[-1] * height * width,
            final_hidden,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.0,
        )

    @staticmethod
    def make_block(channel):
        block_layers = [
            Swish(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
            Swish(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
        ]
        return block_layers

    @staticmethod
    def forward_block(input_hidden, block_layers):
        hidden = input_hidden
        for layer in block_layers:
            hidden = layer(hidden)
        return hidden + input_hidden

    def forward(self, visual_obs):
        batch_size = visual_obs.shape[0]
        hidden = visual_obs
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                hidden = layer(hidden)
            elif isinstance(layer, list):
                hidden = self.forward_block(hidden, layer)
        before_out = hidden.view(batch_size, -1)
        return torch.relu(self.dense(before_out))

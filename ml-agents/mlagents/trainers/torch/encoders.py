import torch
from torch import nn


class VectorEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        self.seq_layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.seq_layers(inputs)


class Normalizer(nn.Module):
    def __init__(self, vec_obs_size, **kwargs):
        super().__init__(**kwargs)
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


def pool_out_shape(h_w, kernel_size):
    height = (h_w[0] - kernel_size) // 2 + 1
    width = (h_w[1] - kernel_size) // 2 + 1
    return height, width


class SimpleVisualEncoder(nn.Module):
    def __init__(self, height, width, initial_channels, output_size):
        super().__init__()
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
        # hidden = torch.relu(self.dense(conv_2.view([-1, self.final_flat])))
        hidden = torch.relu(self.dense(torch.reshape(conv_2, (-1, self.final_flat))))
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
        self.dense = nn.Linear(self.final_flat, self.h_size)

    def forward(self, visual_obs):
        conv_1 = torch.relu(self.conv1(visual_obs))
        conv_2 = torch.relu(self.conv2(conv_1))
        conv_3 = torch.relu(self.conv3(conv_2))
        hidden = torch.relu(self.dense(conv_3.view([-1, self.final_flat])))
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
        self.layers.append(nn.ReLU())
        self.dense = nn.Linear(n_channels[-1] * height * width, final_hidden)

    @staticmethod
    def make_block(channel):
        block_layers = [
            nn.ReLU(),
            nn.Conv2d(channel, channel, [3, 3], [1, 1], padding=1),
            nn.ReLU(),
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

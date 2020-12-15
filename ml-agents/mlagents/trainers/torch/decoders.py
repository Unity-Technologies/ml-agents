from typing import List, Dict

from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch.layers import (
    linear_layer,
    LinearEncoder,
    Initialization,
    Swish,
)

from collections import defaultdict


class ValueHeads(nn.Module):
    def __init__(self, stream_names: List[str], input_size: int, output_size: int = 1):
        super().__init__()
        self.stream_names = stream_names
        _value_heads = {}

        for name in stream_names:
            value = linear_layer(input_size, output_size)
            _value_heads[name] = value
        self.value_heads = nn.ModuleDict(_value_heads)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        value_outputs = {}
        for stream_name, head in self.value_heads.items():
            value_outputs[stream_name] = head(hidden).squeeze(-1)
        return value_outputs


class HyperNetwork(nn.Module):
    def __init__(self, input_size, output_size, hyper_input_size, num_layers, layer_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        layers = [linear_layer(
            hyper_input_size,
            layer_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.0,
            bias_init=Initialization.Zero,
        ), Swish()]
        for _ in range(num_layers - 1):
            layers.append(
                linear_layer(
                    layer_size,
                    layer_size,
                    kernel_init=Initialization.KaimingHeNormal,
                    kernel_gain=1.0,
                    bias_init=Initialization.Zero,
                )
            )
            layers.append(Swish())
        flat_output = linear_layer(
            layer_size,
            input_size * output_size + output_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=0.1,
            bias_init=Initialization.Zero,
        )
        self.hypernet = torch.nn.Sequential(*layers, flat_output)

    def forward(self, input_activation, hyper_input):
        flat_output_weights = self.hypernet(hyper_input)
        batch_size = input_activation.size(0)

        output_weights, output_bias = torch.split(
            flat_output_weights,
            self.input_size * self.output_size,
            dim=-1,
        )

        output_weights = output_weights.view(batch_size, self.input_size, self.output_size)
        output_bias = output_bias.view(batch_size, self.output_size)
        print(output_weights.shape, output_bias.shape, input_activation.shape)
        output = torch.bmm(input_activation.unsqueeze(1), output_weights).squeeze(1) + output_bias
        print(output.shape)
        return output


class ValueHeadsHyperNetwork(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_size,
        goal_size,
        stream_names: List[str],
        input_size: int,
        output_size: int = 1,
    ):
        super().__init__()
        self.stream_names = stream_names
        self._num_goals = goal_size
        self.input_size = input_size
        self.output_size = output_size
        self.streams_size = len(stream_names)
        self.hypernetwork = HyperNetwork(input_size, self.output_size * self.streams_size, goal_size, num_layers, layer_size)

    def forward(
        self, hidden: torch.Tensor, goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output = self.hypernetwork(hidden, goal)
        value_outputs = {}
        output_list = torch.split(output, self.output_size, dim=1)
        for stream_name, output_activation in zip(self.stream_names, output_list):
            value_outputs[stream_name] = output_activation
        return value_outputs

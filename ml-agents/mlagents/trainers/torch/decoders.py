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


class ValueHeadsHyperNetwork(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_size,
        num_goals,
        stream_names: List[str],
        input_size: int,
        output_size: int = 1,
    ):
        super().__init__()
        self.stream_names = stream_names
        self._num_goals = num_goals
        self.input_size = input_size
        self.output_size = output_size
        self.streams_size = len(stream_names)
        hypernet_encoder = linear_layer(
            num_goals,
            layer_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=1.0,
            bias_init=Initialization.Zero,
        )

        flat_output = linear_layer(
            layer_size,
            input_size * output_size * self.streams_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=0.1,
            bias_init=Initialization.Zero,
        )
        self.hypernet = torch.nn.Sequential(hypernet_encoder, Swish(), flat_output)

    def forward(
        self, hidden: torch.Tensor, goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        goal_onehot = torch.nn.functional.one_hot(
            goal[0].long(), self._num_goals
        ).float()
        flat_output_weights = self.hypernet(goal_onehot)
        b = hidden.size(0)

        output_heads = torch.reshape(
            flat_output_weights,
            (self.streams_size, b, self.input_size, self.output_size),
        )
        value_outputs = {}
        for stream_name, out_h in zip(self.stream_names, output_heads):
            value_outputs[stream_name] = torch.bmm(
                hidden.unsqueeze(dim=1), out_h
            ).squeeze()
        return value_outputs

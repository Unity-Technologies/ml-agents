from typing import List, Dict

from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch.layers import linear_layer, HyperNetwork


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
        self.hypernetwork = HyperNetwork(
            input_size,
            self.output_size * self.streams_size,
            goal_size,
            num_layers,
            layer_size,
        )

    def forward(
        self, hidden: torch.Tensor, goal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output = self.hypernetwork(hidden, goal)
        value_outputs = {}
        output_list = torch.split(output, self.output_size, dim=1)
        for stream_name, output_activation in zip(self.stream_names, output_list):
            value_outputs[stream_name] = output_activation
        return value_outputs

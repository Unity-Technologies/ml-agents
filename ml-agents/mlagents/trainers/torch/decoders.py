from typing import List, Dict

from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch.layers import linear_layer


class ValueHeads(nn.Module):
    def __init__(self, stream_names: List[str], input_size: int, output_size: int = 1, n_modes: int = 8):
        super().__init__()
        self.stream_names = stream_names
        _value_heads = {}
        self.n_modes = n_modes

        for name in stream_names:
            value = linear_layer(input_size, output_size * n_modes)
            _value_heads[name] = value
        self.value_heads = nn.ModuleDict(_value_heads)

    def forward(self, hidden: torch.Tensor, mode_oh) -> Dict[str, torch.Tensor]:
        value_outputs = {}
        for stream_name, head in self.value_heads.items():
            masked_head = (head(hidden).reshape(-1, 1, self.n_modes) * mode_oh.reshape(-1, 1, self.n_modes)).sum(dim=2)
            value_outputs[stream_name] = masked_head.squeeze(-1)
        return value_outputs

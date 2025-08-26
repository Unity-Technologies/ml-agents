import torch.nn as nn
import torch

_ACTS = {          # mapeia string → classe PyTorch
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
}

class DroneBody(nn.Module):
    def __init__(self, in_size: int, cfg):
        super().__init__()

        self.memory_size = 0 

        layers = []
        last = in_size
        for _ in range(cfg.num_layers):
            layers += [nn.Linear(last, cfg.hidden_units), nn.ReLU()]
            # if cfg.use_layer_norm:
            #     layers += [nn.LayerNorm(cfg.hidden_units)]
            last = cfg.hidden_units
        self.seq = nn.Sequential(*layers)

    def forward(
        self,
        vector_obs,
        visual_obs=None,
        memories=None,
        sequence_length=None,
        **kwargs
    ):
        if isinstance(vector_obs, (list, tuple)):
            vector_obs = torch.cat(vector_obs, dim=-1)
        
        if vector_obs.dtype != torch.float32:
            vector_obs = vector_obs.float()
        
        encoding = self.seq(vector_obs)
        return encoding, memories
    
    def update_normalization(self, buffer):
        return
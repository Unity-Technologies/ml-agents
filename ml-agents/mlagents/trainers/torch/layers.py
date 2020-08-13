import torch
from enum import Enum


class Swish(torch.nn.Module):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mul(data, torch.sigmoid(data))


class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4


_init_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
}


def linear_layer(
    input_size: int,
    output_size: int,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    kernel_gain: float = 1.0,
    bias_init: Initialization = Initialization.Zero,
) -> torch.nn.Module:
    """
    Creates a torch.nn.Linear module and initializes its weights.
    :param input_size: The size of the input tensor
    :param output_size: The size of the output tensor
    :param kernel_init: The Initialization to use for the weights of the layer
    :param kernel_gain: The multiplier for the weights of the kernel. Note that in
    TensorFlow, calling variance_scaling with scale 0.01 is equivalent to calling
    KaimingHeNormal with kernel_gain of 0.1
    :param bias_init: The Initialization to use for the weights of the bias layer
    """
    layer = torch.nn.Linear(input_size, output_size)
    _init_methods[kernel_init](layer.weight.data)
    layer.weight.data *= kernel_gain
    _init_methods[bias_init](layer.bias.data)
    return layer


def lstm_layer(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    batch_first: bool = True,
    forget_bias: float = 1.0,
    kernel_init: Initialization = Initialization.XavierGlorotUniform,
    bias_init: Initialization = Initialization.Zero,
) -> torch.nn.Module:
    """
    Creates a torch.nn.LSTM and initializes its weights and biases. Provides a
    forget_bias offset like is done in TensorFlow.
    """
    lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
    # Add forget_bias to forget gate bias
    for name, param in lstm.named_parameters():
        if "weight" in name:
            _init_methods[kernel_init](param.data)
        elif "bias" in name:
            _init_methods[bias_init](param.data)
            param.data[hidden_size : 2 * hidden_size].add_(forget_bias)
    return lstm


class AMRLMax(torch.nn.Module):
    """
    Implements Aggregation for LSTM as described here:
    https://www.microsoft.com/en-us/research/publication/amrl-aggregated-memory-for-reinforcement-learning/
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        forget_bias: float = 1.0,
        kernel_init: Initialization = Initialization.XavierGlorotUniform,
        bias_init: Initialization = Initialization.Zero,
        num_post_layers: int = 1,
    ):
        super().__init__()
        self.lstm = lstm_layer(
            input_size,
            hidden_size,
            num_layers,
            batch_first,
            forget_bias,
            kernel_init,
            bias_init,
        )
        self.hidden_size = hidden_size
        self.layers = []
        for _ in range(num_post_layers):
            self.layers.append(
                linear_layer(
                    hidden_size,
                    hidden_size,
                    kernel_init=Initialization.KaimingHeNormal,
                    kernel_gain=1.0,
                )
            )
            self.layers.append(Swish())
        self.seq_layers = torch.nn.Sequential(*self.layers)

    def forward(self, input_tensor, h0_c0):
        hidden = h0_c0
        all_c = []
        m = None
        lstm_out, hidden = self.lstm(input_tensor, hidden)
        h_half, other_half = torch.split(lstm_out, self.hidden_size // 2, dim=-1)
        for t in range(h_half.shape[1]):
            h_half_subt = h_half[:, t : t + 1, :]
            if m is None:
                m = h_half_subt
            else:
                m = torch.max(m, h_half_subt)
            all_c.append(m)
        concat_c = torch.cat(all_c, dim=1)
        concat_out = torch.cat([concat_c, other_half], dim=-1)
        full_out = self.seq_layers(concat_out.reshape([-1, self.hidden_size]))
        full_out = full_out.reshape([-1, input_tensor.shape[1], self.hidden_size])
        return concat_out, hidden

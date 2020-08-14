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
        # Each weight and bias is a concatenation of 4 matrices
        if "weight" in name:
            for idx in range(4):
                block_size = param.shape[0] // 4
                _init_methods[kernel_init](
                    param.data[idx * block_size : (idx + 1) * block_size]
                )
        if "bias" in name:
            for idx in range(4):
                block_size = param.shape[0] // 4
                _init_methods[bias_init](
                    param.data[idx * block_size : (idx + 1) * block_size]
                )
                if idx == 1:
                    param.data[idx * block_size : (idx + 1) * block_size].add_(
                        forget_bias
                    )
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

    @property
    def memory_size(self) -> int:
        return self.hidden_size // 2 + 2 * self.hidden_size

    def forward(self, input_tensor, memories):
        # memories is 1/2 * hidden_size (accumulant) + hidden_size/2 (h0) + hidden_size/2 (c0)
        acc, h0, c0 = torch.split(
            memories,
            [self.hidden_size // 2, self.hidden_size, self.hidden_size],
            dim=-1,
        )
        hidden = (h0, c0)
        all_c = []
        m = acc.permute([1, 0, 2])
        lstm_out, (h0_out, c0_out) = self.lstm(input_tensor, hidden)
        h_half, other_half = torch.split(lstm_out, self.hidden_size // 2, dim=-1)
        for t in range(h_half.shape[1]):
            h_half_subt = h_half[:, t : t + 1, :]
            m = AMRLMax.PassthroughMax.apply(m, h_half_subt)
            all_c.append(m)
        concat_c = torch.cat(all_c, dim=1)
        concat_out = torch.cat([concat_c, other_half], dim=-1)
        full_out = self.seq_layers(concat_out.reshape([-1, self.hidden_size]))
        full_out = full_out.reshape([-1, input_tensor.shape[1], self.hidden_size])
        output_mem = torch.cat([m.permute([1, 0, 2]), h0_out, c0_out], dim=-1)
        return concat_out, output_mem

    class PassthroughMax(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor1, tensor2):
            return torch.max(tensor1, tensor2)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.clone(), grad_output.clone()

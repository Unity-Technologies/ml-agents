from mlagents.torch_utils import torch
import abc
import math
from typing import Tuple
from enum import Enum

from mlagents.trainers.settings import ConditioningType


class Swish(torch.nn.Module):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mul(data, torch.sigmoid(data))


class Initialization(Enum):
    Zero = 0
    XavierGlorotNormal = 1
    XavierGlorotUniform = 2
    KaimingHeNormal = 3  # also known as Variance scaling
    KaimingHeUniform = 4
    Normal = 5


_init_methods = {
    Initialization.Zero: torch.zero_,
    Initialization.XavierGlorotNormal: torch.nn.init.xavier_normal_,
    Initialization.XavierGlorotUniform: torch.nn.init.xavier_uniform_,
    Initialization.KaimingHeNormal: torch.nn.init.kaiming_normal_,
    Initialization.KaimingHeUniform: torch.nn.init.kaiming_uniform_,
    Initialization.Normal: torch.nn.init.normal_,
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
    TensorFlow, the gain is square-rooted. Therefore calling  with scale 0.01 is equivalent to calling
        KaimingHeNormal with kernel_gain of 0.1
    :param bias_init: The Initialization to use for the weights of the bias layer
    """
    layer = torch.nn.Linear(input_size, output_size)
    if (
        kernel_init == Initialization.KaimingHeNormal
        or kernel_init == Initialization.KaimingHeUniform
    ):
        _init_methods[kernel_init](layer.weight.data, nonlinearity="linear")
    else:
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


class MemoryModule(torch.nn.Module):
    @abc.abstractproperty
    def memory_size(self) -> int:
        """
        Size of memory that is required at the start of a sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
        self, input_tensor: torch.Tensor, memories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass a sequence to the memory module.
        :input_tensor: Tensor of shape (batch_size, seq_length, size) that represents the input.
        :memories: Tensor of initial memories.
        :return: Tuple of output, final memories.
        """
        pass


class LayerNorm(torch.nn.Module):
    """
    A vanilla implementation of layer normalization  https://arxiv.org/pdf/1607.06450.pdf
    norm_x = (x - mean) / sqrt((x - mean) ^ 2)
    This does not include the trainable parameters gamma and beta for performance speed.
    Typically, this is norm_x * gamma + beta
    """

    def forward(self, layer_activations: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(layer_activations, dim=-1, keepdim=True)
        var = torch.mean((layer_activations - mean) ** 2, dim=-1, keepdim=True)
        return (layer_activations - mean) / (torch.sqrt(var + 1e-5))


class ConditionalLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        goal_size: int,
        num_layers: int,
        hidden_size: int,
        condition_type: ConditioningType,
    ):
        super().__init__()
        self.condition_type = condition_type
        if self.condition_type != ConditioningType.HYPER:
            self.goal_encoder = LinearEncoder(
                goal_size, num_layers, hidden_size, final_activation=True
            )
            self.input_encoder = linear_layer(input_size, hidden_size)
        else:
            self.input_encoder = HyperNetwork(
                input_size, hidden_size, goal_size, num_layers, hidden_size
            )

    def forward(
        self, input_tensor: torch.Tensor, goal_tensor: torch.Tensor
    ) -> torch.Tensor:
        if self.condition_type == ConditioningType.SOFT_MUL:
            return self.input_encoder(input_tensor) * self.goal_encoder(goal_tensor)
        elif self.condition_type == ConditioningType.SOFT_SUM:
            return self.input_encoder(input_tensor) + self.goal_encoder(goal_tensor)
        else:
            return self.input_encoder(input_tensor, goal_tensor)


class ConditionalEncoder(torch.nn.Module):
    """
    Linear layers.
    """

    def __init__(
        self,
        input_size: int,
        goal_size: int,
        num_layers: int,
        hidden_size: int,
        condition_type: ConditioningType,
        conditional_layers: int,
        kernel_init: Initialization = Initialization.KaimingHeNormal,
        kernel_gain: float = 1.0,
    ):
        super().__init__()
        self.layers = []
        prev_size = input_size + goal_size
        for i in range(num_layers):
            if num_layers - i <= conditional_layers:
                self.layers.append(
                    ConditionalLayer(
                        prev_size, goal_size, 2, hidden_size, condition_type
                    )
                )
            else:
                self.layers.append(
                    linear_layer(
                        prev_size,
                        hidden_size,
                        kernel_init=kernel_init,
                        kernel_gain=kernel_gain,
                    )
                )
            self.layers.append(Swish())
            prev_size = hidden_size
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(
        self, input_tensor: torch.Tensor, goal_tensor: torch.Tensor
    ) -> torch.Tensor:
        activation = torch.cat([input_tensor, goal_tensor], dim=-1)
        for layer in self.layers:
            if isinstance(layer, ConditionalLayer):
                activation = layer(activation, goal_tensor)
            else:
                activation = layer(activation)
        return activation


class LinearEncoder(torch.nn.Module):
    """
    Linear layers.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        kernel_init: Initialization = Initialization.KaimingHeNormal,
        kernel_gain: float = 1.0,
        final_activation: bool = True,
    ):
        super().__init__()
        self.layers = [
            linear_layer(
                input_size,
                hidden_size,
                kernel_init=kernel_init,
                kernel_gain=kernel_gain,
            )
        ]
        self.layers.append(Swish())
        for i in range(num_layers - 1):
            self.layers.append(
                linear_layer(
                    hidden_size,
                    hidden_size,
                    kernel_init=kernel_init,
                    kernel_gain=kernel_gain,
                )
            )
            if i < num_layers - 2 or final_activation:
                self.layers.append(Swish())
        self.seq_layers = torch.nn.Sequential(*self.layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.seq_layers(input_tensor)


class LSTM(MemoryModule):
    """
    Memory module that implements LSTM.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int,
        num_layers: int = 1,
        forget_bias: float = 1.0,
        kernel_init: Initialization = Initialization.XavierGlorotUniform,
        bias_init: Initialization = Initialization.Zero,
    ):
        super().__init__()
        # We set hidden size to half of memory_size since the initial memory
        # will be divided between the hidden state and initial cell state.
        self.hidden_size = memory_size // 2
        self.lstm = lstm_layer(
            input_size,
            self.hidden_size,
            num_layers,
            True,
            forget_bias,
            kernel_init,
            bias_init,
        )

    @property
    def memory_size(self) -> int:
        return 2 * self.hidden_size

    def forward(
        self, input_tensor: torch.Tensor, memories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We don't use torch.split here since it is not supported by Barracuda
        h0 = memories[:, :, : self.hidden_size].contiguous()
        c0 = memories[:, :, self.hidden_size :].contiguous()
        hidden = (h0, c0)
        lstm_out, hidden_out = self.lstm(input_tensor, hidden)
        output_mem = torch.cat(hidden_out, dim=-1)
        return lstm_out, output_mem


class HyperNetwork(torch.nn.Module):
    def __init__(
        self, input_size, output_size, hyper_input_size, num_layers, layer_size
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layer_in_size = hyper_input_size
        layers = []
        for _ in range(num_layers):
            layers.append(
                linear_layer(
                    layer_in_size,
                    layer_size,
                    kernel_init=Initialization.KaimingHeNormal,
                    kernel_gain=1.0,
                    bias_init=Initialization.Zero,
                )
            )
            layers.append(Swish())
            layer_in_size = layer_size
        flat_output = linear_layer(
            layer_size,
            input_size * output_size,
            kernel_init=Initialization.KaimingHeNormal,
            kernel_gain=0.1,
            bias_init=Initialization.Zero,
        )

        # Re-initializing the weights of the last layer of the hypernetwork
        bound = math.sqrt(1 / (layer_size * self.input_size))
        flat_output.weight.data.uniform_(-bound, bound)

        self.hypernet = torch.nn.Sequential(*layers, LayerNorm(), flat_output)

        # The hypernetwork will not generate the bias of the main network layer
        self.bias = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, input_activation, hyper_input):
        output_weights = self.hypernet(hyper_input)

        output_weights = output_weights.view(-1, self.input_size, self.output_size)
        return (
            torch.bmm(input_activation.unsqueeze(1), output_weights).squeeze(1)
            + self.bias
        )

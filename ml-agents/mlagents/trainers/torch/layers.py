from mlagents.torch_utils import torch
import abc
from typing import Tuple, List, Optional
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


class LinearEncoder(torch.nn.Module):
    """
    Linear layers.
    """

    def __init__(self, input_size: int, num_layers: int, hidden_size: int):
        super().__init__()
        self.layers = [
            linear_layer(
                input_size,
                hidden_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.0,
            )
        ]
        self.layers.append(Swish())
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
        h0 = memories[:, :, : self.hidden_size]
        c0 = memories[:, :, self.hidden_size :]
        hidden = (h0, c0)
        lstm_out, hidden_out = self.lstm(input_tensor, hidden)
        output_mem = torch.cat(hidden_out, dim=-1)
        return lstm_out, output_mem


class MultiHeadAttention(torch.nn.Module):
    """
    Multi Head Attention module. We do not use the regular Torch implementation since
    Barracuda does not support some operators it uses.
    Takes as input to the forward method 3 tensors:
     - query: of dimensions (batch_size, number_of_queries, key_size)
     - key: of dimensions (batch_size, number_of_keys, key_size)
     - value: of dimensions (batch_size, number_of_keys, value_size)
    The forward method will return 2 tensors:
     - The output: (batch_size, number_of_queries, output_size)
     - The attention matrix: (batch_size, num_heads, number_of_queries, number_of_keys)
    """

    NEG_INF = -1e6

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        output_size: int,
        num_heads: int,
        embedding_size: int,
    ):
        super().__init__()
        self.n_heads, self.embedding_size = num_heads, embedding_size
        self.output_size = output_size
        self.fc_q = torch.nn.Linear(query_size, self.n_heads * self.embedding_size)
        self.fc_k = torch.nn.Linear(key_size, self.n_heads * self.embedding_size)
        self.fc_v = torch.nn.Linear(value_size, self.n_heads * self.embedding_size)
        # self.fc_q = LinearEncoder(query_size, 2, self.n_heads * self.embedding_size)
        # self.fc_k = LinearEncoder(key_size,2, self.n_heads * self.embedding_size)
        # self.fc_v = LinearEncoder(value_size,2, self.n_heads * self.embedding_size)
        self.fc_out = torch.nn.Linear(
            self.n_heads * self.embedding_size, self.output_size
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
        number_of_keys: int = -1,
        number_of_queries: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = -1  # the batch size
        # This is to avoid using .size() when possible as Barracuda does not support
        n_q = number_of_queries if number_of_queries != -1 else query.size(1)
        n_k = number_of_keys if number_of_keys != -1 else key.size(1)

        # Create a key mask : Only 1 if all values are 0 # shape = (b, n_k)
        # key_mask = torch.sum(key ** 2, axis=2) < 0.01
        key_mask = key_mask.reshape(b, 1, 1, n_k)

        query = self.fc_q(query)  # (b, n_q, h*d)
        key = self.fc_k(key)  # (b, n_k, h*d)
        value = self.fc_v(value)  # (b, n_k, h*d)

        query = query.reshape(b, n_q, self.n_heads, self.embedding_size)
        key = key.reshape(b, n_k, self.n_heads, self.embedding_size)
        value = value.reshape(b, n_k, self.n_heads, self.embedding_size)

        query = query.permute([0, 2, 1, 3])  # (b, h, n_q, emb)
        # The next few lines are equivalent to : key.permute([0, 2, 3, 1])
        # This is a hack, ONNX will compress two permute operations and
        # Barracuda will not like seeing `permute([0,2,3,1])`
        key = key.permute([0, 2, 1, 3])  # (b, h, emb, n_k)
        key -= 1
        key += 1
        key = key.permute([0, 1, 3, 2])  # (b, h, emb, n_k)

        qk = torch.matmul(query, key)  # (b, h, n_q, n_k)

        qk = (1 - key_mask) * qk / (self.embedding_size ** 0.5) + key_mask * self.NEG_INF

        att = torch.softmax(qk, dim=3)  # (b, h, n_q, n_k)

        value = value.permute([0, 2, 1, 3])  # (b, h, n_k, emb)
        value_attention = torch.matmul(att, value)  # (b, h, n_q, emb)

        value_attention = value_attention.permute([0, 2, 1, 3])  # (b, n_q, h, emb)
        value_attention = value_attention.reshape(
            b, n_q, self.n_heads * self.embedding_size
        )  # (b, n_q, h*emb)

        out = self.fc_out(value_attention)  # (b, n_q, emb)
        return out, att


class ZeroObservationMask(torch.nn.Module):
    """
    Takes a List of Tensors and returns a List of mask Tensor with 1 if the input was
    all zeros and 0 otherwise. This is used in the Attention layer to mask the padding
    observations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, observations: List[torch.Tensor]):
        with torch.no_grad():
            # Generate the masking tensors for each entities tensor (mask only if all zeros)
            key_masks: List[torch.Tensor] = [
                (torch.sum(ent ** 2, axis=2) < 0.01).type(torch.FloatTensor)
                for ent in observations
            ]
        return key_masks


class SimpleTransformer(torch.nn.Module):
    """
    A simple architecture inspired from https://arxiv.org/pdf/1909.07528.pdf that uses
    multi head self attention to encode information about a "Self" and a list of
    relevant "Entities".
    """

    EPISLON = 1e-7

    def __init__(
        self, x_self_size: int, entities_sizes: List[int], embedding_size: int
    ):
        super().__init__()
        self.self_size = x_self_size
        self.entities_sizes = entities_sizes
        self.entities_num_max_elements: Optional[List[int]] = None
        self.ent_encoders = torch.nn.ModuleList(
            [
                LinearEncoder(self.self_size + ent_size, 1, embedding_size)
                for ent_size in self.entities_sizes
            ]
        )
        self.attention = MultiHeadAttention(
            query_size=embedding_size,
            key_size=embedding_size,
            value_size=embedding_size,
            output_size=embedding_size,
            num_heads=4,
            embedding_size=embedding_size,
        )
        self.residual_layer = LinearEncoder(embedding_size, 1, embedding_size)

    def forward(self, x_self: torch.Tensor, entities: List[torch.Tensor], key_masks: List[torch.Tensor]):
        # Gather the maximum number of entities information
        if self.entities_num_max_elements is None:
            self.entities_num_max_elements = []
            for ent in entities:
                self.entities_num_max_elements.append(ent.shape[1])
        # Concatenate all observations with self
        self_and_ent: List[torch.Tensor] = []
        for num_entities, ent in zip(self.entities_num_max_elements, entities):
            expanded_self = x_self.reshape(-1, 1, self.self_size)
            # .repeat(
            #     1, num_entities, 1
            # )
            expanded_self = torch.cat([expanded_self] * num_entities, dim=1)
            self_and_ent.append(torch.cat([expanded_self, ent], dim=2))
        # Generate the tensor that will serve as query, key and value to self attention
        qkv = torch.cat(
            [ent_encoder(x) for ent_encoder, x in zip(self.ent_encoders, self_and_ent)],
            dim=1,
        )
        mask = torch.cat(key_masks, dim=1)
        # Feed to self attention
        max_num_ent = sum(self.entities_num_max_elements)
        output, _ = self.attention(qkv, qkv, qkv, mask, max_num_ent, max_num_ent)
        # Residual
        output = self.residual_layer(output) + qkv
        # Average Pooling
        numerator = torch.sum(output * (1 - mask).reshape(-1, max_num_ent, 1), dim=1)
        denominator = torch.sum(1 - mask, dim=1, keepdim=True) + self.EPISLON
        output = numerator / denominator
        return output

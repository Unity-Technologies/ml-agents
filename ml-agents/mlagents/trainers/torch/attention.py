from mlagents.torch_utils import torch
import warnings
from typing import Tuple, Optional, List
from mlagents.trainers.torch.layers import (
    LinearEncoder,
    Initialization,
    linear_layer,
    LayerNorm,
)
from mlagents.trainers.torch.model_serialization import exporting_to_onnx
from mlagents.trainers.exception import UnityTrainerException


def get_zero_entities_mask(entities: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Takes a List of Tensors and returns a List of mask Tensor with 1 if the input was
    all zeros (on dimension 2) and 0 otherwise. This is used in the Attention
    layer to mask the padding observations.
    """
    with torch.no_grad():

        if exporting_to_onnx.is_exporting():
            with warnings.catch_warnings():
                # We ignore a TracerWarning from PyTorch that warns that doing
                # shape[n].item() will cause the trace to be incorrect (the trace might
                # not generalize to other inputs)
                # We ignore this warning because we know the model will always be
                # run with inputs of the same shape
                warnings.simplefilter("ignore")
                # When exporting to ONNX, we want to transpose the entities. This is
                # because ONNX only support input in NCHW (channel first) format.
                # Barracuda also expect to get data in NCHW.
                entities = [
                    torch.transpose(obs, 2, 1).reshape(
                        -1, obs.shape[1].item(), obs.shape[2].item()
                    )
                    for obs in entities
                ]

        # Generate the masking tensors for each entities tensor (mask only if all zeros)
        key_masks: List[torch.Tensor] = [
            (torch.sum(ent ** 2, axis=2) < 0.01).float() for ent in entities
        ]
    return key_masks


class MultiHeadAttention(torch.nn.Module):

    NEG_INF = -1e6

    def __init__(self, embedding_size: int, num_heads: int):
        """
        Multi Head Attention module. We do not use the regular Torch implementation since
        Barracuda does not support some operators it uses.
        Takes as input to the forward method 3 tensors:
        - query: of dimensions (batch_size, number_of_queries, embedding_size)
        - key: of dimensions (batch_size, number_of_keys, embedding_size)
        - value: of dimensions (batch_size, number_of_keys, embedding_size)
        The forward method will return 2 tensors:
        - The output: (batch_size, number_of_queries, embedding_size)
        - The attention matrix: (batch_size, num_heads, number_of_queries, number_of_keys)
        :param embedding_size: The size of the embeddings that will be generated (should be
        dividable by the num_heads)
        :param total_max_elements: The maximum total number of entities that can be passed to
        the module
        :param num_heads: The number of heads of the attention module
        """
        super().__init__()
        self.n_heads = num_heads
        self.head_size: int = embedding_size // self.n_heads
        self.embedding_size: int = self.head_size * self.n_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        n_q: int,
        n_k: int,
        key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = -1  # the batch size

        query = query.reshape(
            b, n_q, self.n_heads, self.head_size
        )  # (b, n_q, h, emb / h)
        key = key.reshape(b, n_k, self.n_heads, self.head_size)  # (b, n_k, h, emb / h)
        value = value.reshape(
            b, n_k, self.n_heads, self.head_size
        )  # (b, n_k, h, emb / h)

        query = query.permute([0, 2, 1, 3])  # (b, h, n_q, emb / h)
        # The next few lines are equivalent to : key.permute([0, 2, 3, 1])
        # This is a hack, ONNX will compress two permute operations and
        # Barracuda will not like seeing `permute([0,2,3,1])`
        key = key.permute([0, 2, 1, 3])  # (b, h, emb / h, n_k)
        key -= 1
        key += 1
        key = key.permute([0, 1, 3, 2])  # (b, h, emb / h, n_k)

        qk = torch.matmul(query, key)  # (b, h, n_q, n_k)

        if key_mask is None:
            qk = qk / (self.embedding_size ** 0.5)
        else:
            key_mask = key_mask.reshape(b, 1, 1, n_k)
            qk = (1 - key_mask) * qk / (
                self.embedding_size ** 0.5
            ) + key_mask * self.NEG_INF

        att = torch.softmax(qk, dim=3)  # (b, h, n_q, n_k)

        value = value.permute([0, 2, 1, 3])  # (b, h, n_k, emb / h)
        value_attention = torch.matmul(att, value)  # (b, h, n_q, emb / h)

        value_attention = value_attention.permute([0, 2, 1, 3])  # (b, n_q, h, emb / h)
        value_attention = value_attention.reshape(
            b, n_q, self.embedding_size
        )  # (b, n_q, emb)

        return value_attention, att


class EntityEmbedding(torch.nn.Module):
    """
    A module used to embed entities before passing them to a self-attention block.
    Used in conjunction with ResidualSelfAttention to encode information about a self
    and additional entities. Can also concatenate self to entities for ego-centric self-
    attention. Inspired by architecture used in https://arxiv.org/pdf/1909.07528.pdf.
    """

    def __init__(
        self,
        entity_size: int,
        entity_num_max_elements: Optional[int],
        embedding_size: int,
    ):
        """
        Constructs an EntityEmbedding module.
        :param x_self_size: Size of "self" entity.
        :param entity_size: Size of other entities.
        :param entity_num_max_elements: Maximum elements for a given entity, None for unrestricted.
            Needs to be assigned in order for model to be exportable to ONNX and Barracuda.
        :param embedding_size: Embedding size for the entity encoder.
        :param concat_self: Whether to concatenate x_self to entities. Set True for ego-centric
            self-attention.
        """
        super().__init__()
        self.self_size: int = 0
        self.entity_size: int = entity_size
        self.entity_num_max_elements: int = -1
        if entity_num_max_elements is not None:
            self.entity_num_max_elements = entity_num_max_elements
        self.embedding_size = embedding_size
        # Initialization scheme from http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        self.self_ent_encoder = LinearEncoder(
            self.entity_size,
            1,
            self.embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / self.embedding_size) ** 0.5,
        )

    def add_self_embedding(self, size: int) -> None:
        self.self_size = size
        self.self_ent_encoder = LinearEncoder(
            self.self_size + self.entity_size,
            1,
            self.embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / self.embedding_size) ** 0.5,
        )

    def forward(self, x_self: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        num_entities = self.entity_num_max_elements
        if num_entities < 0:
            if exporting_to_onnx.is_exporting():
                raise UnityTrainerException(
                    "Trying to export an attention mechanism that doesn't have a set max \
                    number of elements."
                )
            num_entities = entities.shape[1]

        if exporting_to_onnx.is_exporting():
            # When exporting to ONNX, we want to transpose the entities. This is
            # because ONNX only support input in NCHW (channel first) format.
            # Barracuda also expect to get data in NCHW.
            entities = torch.transpose(entities, 2, 1).reshape(
                -1, num_entities, self.entity_size
            )

        if self.self_size > 0:
            expanded_self = x_self.reshape(-1, 1, self.self_size)
            expanded_self = torch.cat([expanded_self] * num_entities, dim=1)
            # Concatenate all observations with self
            entities = torch.cat([expanded_self, entities], dim=2)
        # Encode entities
        encoded_entities = self.self_ent_encoder(entities)
        return encoded_entities


class ResidualSelfAttention(torch.nn.Module):
    """
    Residual self attentioninspired from https://arxiv.org/pdf/1909.07528.pdf. Can be used
    with an EntityEmbedding module, to apply multi head self attention to encode information
    about a "Self" and a list of relevant "Entities".
    """

    EPSILON = 1e-7

    def __init__(
        self,
        embedding_size: int,
        entity_num_max_elements: Optional[int] = None,
        num_heads: int = 4,
    ):
        """
        Constructs a ResidualSelfAttention module.
        :param embedding_size: Embedding sizee for attention mechanism and
            Q, K, V encoders.
        :param entity_num_max_elements: A List of ints representing the maximum number
            of elements in an entity sequence. Should be of length num_entities. Pass None to
            not restrict the number of elements; however, this will make the module
            unexportable to ONNX/Barracuda.
        :param num_heads: Number of heads for Multi Head Self-Attention
        """
        super().__init__()
        self.max_num_ent: Optional[int] = None
        if entity_num_max_elements is not None:
            self.max_num_ent = entity_num_max_elements

        self.attention = MultiHeadAttention(
            num_heads=num_heads, embedding_size=embedding_size
        )

        # Initialization scheme from http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        self.fc_q = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.fc_k = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.fc_v = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.fc_out = linear_layer(
            embedding_size,
            embedding_size,
            kernel_init=Initialization.Normal,
            kernel_gain=(0.125 / embedding_size) ** 0.5,
        )
        self.embedding_norm = LayerNorm()
        self.residual_norm = LayerNorm()

    def forward(self, inp: torch.Tensor, key_masks: List[torch.Tensor]) -> torch.Tensor:
        # Gather the maximum number of entities information
        mask = torch.cat(key_masks, dim=1)

        inp = self.embedding_norm(inp)
        # Feed to self attention
        query = self.fc_q(inp)  # (b, n_q, emb)
        key = self.fc_k(inp)  # (b, n_k, emb)
        value = self.fc_v(inp)  # (b, n_k, emb)

        # Only use max num if provided
        if self.max_num_ent is not None:
            num_ent = self.max_num_ent
        else:
            num_ent = inp.shape[1]
            if exporting_to_onnx.is_exporting():
                raise UnityTrainerException(
                    "Trying to export an attention mechanism that doesn't have a set max \
                    number of elements."
                )

        output, _ = self.attention(query, key, value, num_ent, num_ent, mask)
        # Residual
        output = self.fc_out(output) + inp
        output = self.residual_norm(output)
        # Average Pooling
        numerator = torch.sum(output * (1 - mask).reshape(-1, num_ent, 1), dim=1)
        denominator = torch.sum(1 - mask, dim=1, keepdim=True) + self.EPSILON
        output = numerator / denominator
        return output

from mlagents.torch_utils import torch
from typing import Tuple, Optional, List
from mlagents.trainers.torch.layers import LinearEncoder, linear_layer, Initialization


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
        key_mask: Optional[torch.Tensor] = None,
        number_of_keys: int = -1,
        number_of_queries: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = -1  # the batch size
        # This is to avoid using .size() when possible as Barracuda does not support
        n_q = number_of_queries if number_of_queries != -1 else query.size(1)
        n_k = number_of_keys if number_of_keys != -1 else key.size(1)

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

        if key_mask is None:
            qk = qk / (self.embedding_size ** 0.5)
        else:
            key_mask = key_mask.reshape(b, 1, 1, n_k)
            qk = (1 - key_mask) * qk / (
                self.embedding_size ** 0.5
            ) + key_mask * self.NEG_INF

        att = torch.softmax(qk, dim=3)  # (b, h, n_q, n_k)

        value = value.permute([0, 2, 1, 3])  # (b, h, n_k, emb)
        value_attention = torch.matmul(att, value)  # (b, h, n_q, emb)

        value_attention = value_attention.permute([0, 2, 1, 3])  # (b, n_q, h, emb)
        value_attention = value_attention.reshape(
            b, n_q, self.n_heads * self.embedding_size
        )  # (b, n_q, h*emb)

        out = self.fc_out(value_attention)  # (b, n_q, emb)
        return out, att


class SimpleTransformer(torch.nn.Module):
    """
    A simple architecture inspired from https://arxiv.org/pdf/1909.07528.pdf that uses
    multi head self attention to encode information about a "Self" and a list of
    relevant "Entities".
    """

    EPISLON = 1e-7

    def __init__(
        self,
        x_self_size: int,
        entities_sizes: List[int],
        embedding_size: int,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        self.self_size = x_self_size
        self.entities_sizes = entities_sizes
        self.entities_num_max_elements: Optional[List[int]] = None
        self.ent_encoders = torch.nn.ModuleList(
            [
                # LinearEncoder(self.self_size + ent_size, 2, embedding_size)
                # from http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
                # linear_layer(self.self_size + ent_size, embedding_size, Initialization.Normal, kernel_gain=1 / (self.self_size + ent_size) ** 0.5)
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
        if output_size is None:
            output_size = embedding_size

    def forward(
        self,
        x_self: torch.Tensor,
        entities: List[torch.Tensor],
        key_masks: List[torch.Tensor],
    ) -> torch.Tensor:
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
        # Residual between x_self and the output of the module
        output = torch.cat([output, x_self], dim=1)
        return output

    @staticmethod
    def get_masks(observations: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Takes a List of Tensors and returns a List of mask Tensor with 1 if the input was
        all zeros (on dimension 2) and 0 otherwise. This is used in the Attention
        layer to mask the padding observations.
        """
        with torch.no_grad():
            # Generate the masking tensors for each entities tensor (mask only if all zeros)
            key_masks: List[torch.Tensor] = [
                (torch.sum(ent ** 2, axis=2) < 0.01).type(torch.FloatTensor)
                for ent in observations
            ]
        return key_masks

class SmallestAttention(torch.nn.Module):
    def __init__(
        self,
        x_self_size: int,
        entities_sizes: List[int],
        embedding_size: int,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        self.self_size = x_self_size
        self.entities_sizes = entities_sizes
        self.entities_num_max_elements: Optional[List[int]] = None
        self.ent_encoders = torch.nn.ModuleList(
            [
                LinearEncoder(self.self_size + ent_size, 2, embedding_size)
                # LinearEncoder(self.self_size + ent_size, 3, embedding_size)
                # LinearEncoder(self.self_size + ent_size, 1, embedding_size)
                for ent_size in self.entities_sizes
            ]
        )
        self.importance_layer = LinearEncoder(embedding_size, 1, 1)

    def forward(
        self,
        x_self: torch.Tensor,
        entities: List[torch.Tensor],
        key_masks: List[torch.Tensor],
    ) -> torch.Tensor:
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

        importance = self.importance_layer(qkv) + mask.unsqueeze(2) * -1e6
        importance = torch.softmax(importance, dim=1)
        weighted_qkv = qkv * importance

        output = torch.sum(weighted_qkv, dim=1)
        output = torch.cat([output, x_self], dim=1)

        return output

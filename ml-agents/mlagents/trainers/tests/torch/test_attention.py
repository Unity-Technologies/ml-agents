import pytest
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.layers import linear_layer, LinearEncoder
from mlagents.trainers.torch.attention import (
    MultiHeadAttention,
    EntityEmbedding,
    ResidualSelfAttention,
    get_zero_entities_mask,
)


def test_multi_head_attention_initialization():
    n_h, emb_size = 4, 12
    n_k, n_q, b = 13, 14, 15
    mha = MultiHeadAttention(emb_size, n_h)

    query = torch.ones((b, n_q, emb_size))
    key = torch.ones((b, n_k, emb_size))
    value = torch.ones((b, n_k, emb_size))

    output, attention = mha.forward(query, key, value, n_q, n_k)

    assert output.shape == (b, n_q, emb_size)
    assert attention.shape == (b, n_h, n_q, n_k)


def test_multi_head_attention_masking():
    epsilon = 0.0001
    n_h, emb_size = 4, 12
    n_k, n_q, b = 13, 14, 15
    mha = MultiHeadAttention(emb_size, n_h)
    # create a key input with some keys all 0
    query = torch.ones((b, n_q, emb_size))
    key = torch.ones((b, n_k, emb_size))
    value = torch.ones((b, n_k, emb_size))

    mask = torch.zeros((b, n_k))
    for i in range(n_k):
        if i % 3 == 0:
            key[:, i, :] = 0
            mask[:, i] = 1

    _, attention = mha.forward(query, key, value, n_q, n_k, mask)

    for i in range(n_k):
        if i % 3 == 0:
            assert torch.sum(attention[:, :, :, i] ** 2) < epsilon
        else:
            assert torch.sum(attention[:, :, :, i] ** 2) > epsilon


def test_zero_mask_layer():
    batch_size, size = 10, 30

    def generate_input_helper(pattern):
        _input = torch.zeros((batch_size, 0, size))
        for i in range(len(pattern)):
            if i % 2 == 0:
                _input = torch.cat(
                    [_input, torch.rand((batch_size, pattern[i], size))], dim=1
                )
            else:
                _input = torch.cat(
                    [_input, torch.zeros((batch_size, pattern[i], size))], dim=1
                )
        return _input

    masking_pattern_1 = [3, 2, 3, 4]
    masking_pattern_2 = [5, 7, 8, 2]
    input_1 = generate_input_helper(masking_pattern_1)
    input_2 = generate_input_helper(masking_pattern_2)

    masks = get_zero_entities_mask([input_1, input_2])
    assert len(masks) == 2
    masks_1 = masks[0]
    masks_2 = masks[1]
    assert masks_1.shape == (batch_size, sum(masking_pattern_1))
    assert masks_2.shape == (batch_size, sum(masking_pattern_2))
    for i in masking_pattern_1:
        assert masks_1[0, 1] == 0 if i % 2 == 0 else 1
    for i in masking_pattern_2:
        assert masks_2[0, 1] == 0 if i % 2 == 0 else 1


@pytest.mark.parametrize("mask_value", [0, 1])
def test_all_masking(mask_value):
    # We make sure that a mask of all zeros or all ones will not trigger an error
    np.random.seed(1336)
    torch.manual_seed(1336)
    size, n_k, = 3, 5
    embedding_size = 64
    entity_embeddings = EntityEmbedding(size, n_k, embedding_size)
    entity_embeddings.add_self_embedding(size)
    transformer = ResidualSelfAttention(embedding_size, n_k)
    l_layer = linear_layer(embedding_size, size)
    optimizer = torch.optim.Adam(
        list(entity_embeddings.parameters())
        + list(transformer.parameters())
        + list(l_layer.parameters()),
        lr=0.001,
        weight_decay=1e-6,
    )
    batch_size = 20
    for _ in range(5):
        center = torch.rand((batch_size, size))
        key = torch.rand((batch_size, n_k, size))
        with torch.no_grad():
            # create the target : The key closest to the query in euclidean distance
            distance = torch.sum(
                (center.reshape((batch_size, 1, size)) - key) ** 2, dim=2
            )
            argmin = torch.argmin(distance, dim=1)
            target = []
            for i in range(batch_size):
                target += [key[i, argmin[i], :]]
            target = torch.stack(target, dim=0)
            target = target.detach()

        embeddings = entity_embeddings(center, key)
        masks = [torch.ones_like(key[:, :, 0]) * mask_value]
        prediction = transformer.forward(embeddings, masks)
        prediction = l_layer(prediction)
        prediction = prediction.reshape((batch_size, size))
        error = torch.mean((prediction - target) ** 2, dim=1)
        error = torch.mean(error) / 2
        optimizer.zero_grad()
        error.backward()
        optimizer.step()


def test_predict_closest_training():
    np.random.seed(1336)
    torch.manual_seed(1336)
    size, n_k, = 3, 5
    embedding_size = 64
    entity_embeddings = EntityEmbedding(size, n_k, embedding_size)
    entity_embeddings.add_self_embedding(size)
    transformer = ResidualSelfAttention(embedding_size, n_k)
    l_layer = linear_layer(embedding_size, size)
    optimizer = torch.optim.Adam(
        list(entity_embeddings.parameters())
        + list(transformer.parameters())
        + list(l_layer.parameters()),
        lr=0.001,
        weight_decay=1e-6,
    )
    batch_size = 200
    for _ in range(200):
        center = torch.rand((batch_size, size))
        key = torch.rand((batch_size, n_k, size))
        with torch.no_grad():
            # create the target : The key closest to the query in euclidean distance
            distance = torch.sum(
                (center.reshape((batch_size, 1, size)) - key) ** 2, dim=2
            )
            argmin = torch.argmin(distance, dim=1)
            target = []
            for i in range(batch_size):
                target += [key[i, argmin[i], :]]
            target = torch.stack(target, dim=0)
            target = target.detach()

        embeddings = entity_embeddings(center, key)
        masks = get_zero_entities_mask([key])
        prediction = transformer.forward(embeddings, masks)
        prediction = l_layer(prediction)
        prediction = prediction.reshape((batch_size, size))
        error = torch.mean((prediction - target) ** 2, dim=1)
        error = torch.mean(error) / 2
        print(error.item())
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    assert error.item() < 0.02


def test_predict_minimum_training():
    # of 5 numbers, predict index of min
    np.random.seed(1336)
    torch.manual_seed(1336)
    n_k = 5
    size = n_k + 1
    embedding_size = 64
    entity_embedding = EntityEmbedding(size, n_k, embedding_size)  # no self
    transformer = ResidualSelfAttention(embedding_size)
    l_layer = LinearEncoder(embedding_size, 2, n_k)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(entity_embedding.parameters())
        + list(transformer.parameters())
        + list(l_layer.parameters()),
        lr=0.001,
        weight_decay=1e-6,
    )

    batch_size = 200
    onehots = ModelUtils.actions_to_onehot(torch.range(0, n_k - 1).unsqueeze(1), [n_k])[
        0
    ]
    onehots = onehots.expand((batch_size, -1, -1))
    losses = []
    for _ in range(400):
        num = np.random.randint(0, n_k)
        inp = torch.rand((batch_size, num + 1, 1))
        with torch.no_grad():
            # create the target : The minimum
            argmin = torch.argmin(inp, dim=1)
            argmin = argmin.squeeze()
            argmin = argmin.detach()
        sliced_oh = onehots[:, : num + 1]
        inp = torch.cat([inp, sliced_oh], dim=2)

        embeddings = entity_embedding(inp, inp)
        masks = get_zero_entities_mask([inp])
        prediction = transformer(embeddings, masks)
        prediction = l_layer(prediction)
        ce = loss(prediction, argmin)
        losses.append(ce.item())
        print(ce.item())
        optimizer.zero_grad()
        ce.backward()
        optimizer.step()
    assert np.array(losses[-20:]).mean() < 0.1

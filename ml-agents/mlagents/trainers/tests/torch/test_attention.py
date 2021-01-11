from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.layers import linear_layer
from mlagents.trainers.torch.attention import MultiHeadAttention, SimpleTransformer


def test_multi_head_attention_initialization():
    q_size, k_size, v_size, o_size, n_h, emb_size = 7, 8, 9, 10, 11, 12
    n_k, n_q, b = 13, 14, 15
    mha = MultiHeadAttention(q_size, k_size, v_size, o_size, n_h, emb_size)

    query = torch.ones((b, n_q, q_size))
    key = torch.ones((b, n_k, k_size))
    value = torch.ones((b, n_k, v_size))

    output, attention = mha.forward(query, key, value)

    assert output.shape == (b, n_q, o_size)
    assert attention.shape == (b, n_h, n_q, n_k)


def test_multi_head_attention_masking():
    epsilon = 0.0001
    q_size, k_size, v_size, o_size, n_h, emb_size = 7, 8, 9, 10, 11, 12
    n_k, n_q, b = 13, 14, 15
    mha = MultiHeadAttention(q_size, k_size, v_size, o_size, n_h, emb_size)

    # create a key input with some keys all 0
    key = torch.ones((b, n_k, k_size))
    mask = torch.zeros((b, n_k))
    for i in range(n_k):
        if i % 3 == 0:
            key[:, i, :] = 0
            mask[:, i] = 1

    query = torch.ones((b, n_q, q_size))
    value = torch.ones((b, n_k, v_size))

    _, attention = mha.forward(query, key, value, mask)
    for i in range(n_k):
        if i % 3 == 0:
            assert torch.sum(attention[:, :, :, i] ** 2) < epsilon
        else:
            assert torch.sum(attention[:, :, :, i] ** 2) > epsilon


def test_multi_head_attention_training():
    np.random.seed(1336)
    torch.manual_seed(1336)
    size, n_h, n_k, n_q = 3, 10, 5, 1
    embedding_size = 64
    mha = MultiHeadAttention(size, size, size, size, n_h, embedding_size)
    optimizer = torch.optim.Adam(mha.parameters(), lr=0.001)
    batch_size = 200
    point_range = 3
    init_error = -1.0
    for _ in range(50):
        query = torch.rand((batch_size, n_q, size)) * point_range * 2 - point_range
        key = torch.rand((batch_size, n_k, size)) * point_range * 2 - point_range
        value = key
        with torch.no_grad():
            # create the target : The key closest to the query in euclidean distance
            distance = torch.sum((query - key) ** 2, dim=2)
            argmin = torch.argmin(distance, dim=1)
            target = []
            for i in range(batch_size):
                target += [key[i, argmin[i], :]]
            target = torch.stack(target, dim=0)
            target = target.detach()

        prediction, _ = mha.forward(query, key, value)
        prediction = prediction.reshape((batch_size, size))
        error = torch.mean((prediction - target) ** 2, dim=1)
        error = torch.mean(error) / 2
        if init_error == -1.0:
            init_error = error.item()
        else:
            assert error.item() < init_error
        print(error.item())
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    assert error.item() < 0.5


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

    masks = SimpleTransformer.get_masks([input_1, input_2])
    assert len(masks) == 2
    masks_1 = masks[0]
    masks_2 = masks[1]
    assert masks_1.shape == (batch_size, sum(masking_pattern_1))
    assert masks_2.shape == (batch_size, sum(masking_pattern_2))
    for i in masking_pattern_1:
        assert masks_1[0, 1] == 0 if i % 2 == 0 else 1
    for i in masking_pattern_2:
        assert masks_2[0, 1] == 0 if i % 2 == 0 else 1


def test_simple_transformer_training():
    np.random.seed(1336)
    torch.manual_seed(1336)
    size, n_k, = 3, 5
    embedding_size = 64
    transformer = SimpleTransformer(size, [size], embedding_size)
    l_layer = linear_layer(embedding_size, size)
    optimizer = torch.optim.Adam(
        list(transformer.parameters()) + list(l_layer.parameters()), lr=0.001
    )
    batch_size = 200
    point_range = 3
    init_error = -1.0
    for _ in range(100):
        center = torch.rand((batch_size, size)) * point_range * 2 - point_range
        key = torch.rand((batch_size, n_k, size)) * point_range * 2 - point_range
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

        masks = SimpleTransformer.get_masks([key])
        prediction = transformer.forward(center, [key], masks)
        prediction = l_layer(prediction)
        prediction = prediction.reshape((batch_size, size))
        error = torch.mean((prediction - target) ** 2, dim=1)
        error = torch.mean(error) / 2
        if init_error == -1.0:
            init_error = error.item()
        else:
            assert error.item() < init_error
        print(error.item())
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    assert error.item() < 0.3

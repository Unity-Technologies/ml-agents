from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.layers import (
    Swish,
    linear_layer,
    lstm_layer,
    Initialization,
    LSTM,
    MultiHeadAttention,
)


def test_swish():
    layer = Swish()
    input_tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    target_tensor = torch.mul(input_tensor, torch.sigmoid(input_tensor))
    assert torch.all(torch.eq(layer(input_tensor), target_tensor))


def test_initialization_layer():
    torch.manual_seed(0)
    # Test Zero
    layer = linear_layer(
        3, 4, kernel_init=Initialization.Zero, bias_init=Initialization.Zero
    )
    assert torch.all(torch.eq(layer.weight.data, torch.zeros_like(layer.weight.data)))
    assert torch.all(torch.eq(layer.bias.data, torch.zeros_like(layer.bias.data)))


def test_lstm_layer():
    torch.manual_seed(0)
    # Test zero for LSTM
    layer = lstm_layer(
        4, 4, kernel_init=Initialization.Zero, bias_init=Initialization.Zero
    )
    for name, param in layer.named_parameters():
        if "weight" in name:
            assert torch.all(torch.eq(param.data, torch.zeros_like(param.data)))
        elif "bias" in name:
            assert torch.all(
                torch.eq(param.data[4:8], torch.ones_like(param.data[4:8]))
            )


def test_lstm_class():
    torch.manual_seed(0)
    input_size = 12
    memory_size = 64
    batch_size = 8
    seq_len = 16
    lstm = LSTM(input_size, memory_size)

    assert lstm.memory_size == memory_size

    sample_input = torch.ones((batch_size, seq_len, input_size))
    sample_memories = torch.ones((1, batch_size, memory_size))
    out, mem = lstm(sample_input, sample_memories)
    # Hidden size should be half of memory_size
    assert out.shape == (batch_size, seq_len, memory_size // 2)
    assert mem.shape == (1, batch_size, memory_size)


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
    for i in range(n_k):
        if i % 3 == 0:
            key[:, i, :] = 0

    query = torch.ones((b, n_q, q_size))
    value = torch.ones((b, n_k, v_size))

    _, attention = mha.forward(query, key, value)
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

from mlagents.torch_utils import torch

from mlagents.trainers.torch.layers import (
    Swish,
    linear_layer,
    lstm_layer,
    Initialization,
    LSTM,
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

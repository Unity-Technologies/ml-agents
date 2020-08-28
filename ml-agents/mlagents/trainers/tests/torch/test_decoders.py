import pytest
from mlagents.torch_utils import torch

from mlagents.trainers.torch.decoders import ValueHeads


def test_valueheads():
    stream_names = [f"reward_signal_{num}" for num in range(5)]
    input_size = 5
    batch_size = 4

    # Test default 1 value per head
    value_heads = ValueHeads(stream_names, input_size)
    input_data = torch.ones((batch_size, input_size))
    value_out = value_heads(input_data)  # Note: mean value will be removed shortly

    for stream_name in stream_names:
        assert value_out[stream_name].shape == (batch_size,)

    # Test that inputting the wrong size input will throw an error
    with pytest.raises(Exception):
        value_out = value_heads(torch.ones((batch_size, input_size + 2)))

    # Test multiple values per head (e.g. discrete Q function)
    output_size = 4
    value_heads = ValueHeads(stream_names, input_size, output_size)
    input_data = torch.ones((batch_size, input_size))
    value_out = value_heads(input_data)

    for stream_name in stream_names:
        assert value_out[stream_name].shape == (batch_size, output_size)

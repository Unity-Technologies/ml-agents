import torch

from mlagents.trainers.torch.layers import Swish, linear_layer, Initialization


def test_swish():
    layer = Swish()
    input_tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    target_tensor = torch.mul(input_tensor, torch.sigmoid(input_tensor))
    assert torch.all(torch.eq(layer(input_tensor), target_tensor))


def test_initialization_layer():
    torch.manual_seed(0)
    # Test Zero
    layer = linear_layer(3, 4, Initialization.Zero, Initialization.Zero)
    assert torch.all(torch.eq(layer.weight.data, torch.zeros_like(layer.weight.data)))
    assert torch.all(torch.eq(layer.bias.data, torch.zeros_like(layer.bias.data)))

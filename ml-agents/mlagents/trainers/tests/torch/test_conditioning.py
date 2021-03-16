import pytest
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch.layers import linear_layer
from mlagents.trainers.torch.conditioning import ConditionalEncoder


def test_conditional_layer_initialization():
    b, input_size, goal_size, h, num_cond_layers, num_normal_layers = 7, 10, 8, 16, 2, 1
    conditional_enc = ConditionalEncoder(
        input_size, goal_size, h, num_normal_layers + num_cond_layers, num_cond_layers
    )

    input_tensor = torch.ones(b, input_size)
    goal_tensor = torch.ones(b, goal_size)

    output = conditional_enc.forward(input_tensor, goal_tensor)

    assert output.shape == (b, h)


@pytest.mark.parametrize("num_cond_layers", [1, 2, 3])
def test_predict_with_condition(num_cond_layers):
    np.random.seed(1336)
    torch.manual_seed(1336)
    input_size, goal_size, h, num_normal_layers = 10, 1, 16, 1

    conditional_enc = ConditionalEncoder(
        input_size, goal_size, h, num_normal_layers + num_cond_layers, num_cond_layers
    )
    l_layer = linear_layer(h, 1)

    optimizer = torch.optim.Adam(
        list(conditional_enc.parameters()) + list(l_layer.parameters()), lr=0.001
    )
    batch_size = 200
    for _ in range(300):
        input_tensor = torch.rand((batch_size, input_size))
        goal_tensor = (torch.rand((batch_size, goal_size)) > 0.5).float()
        # If the goal is 1: do the sum of the inputs, else, return 0
        target = torch.sum(input_tensor, dim=1, keepdim=True) * goal_tensor
        target.detach()
        prediction = l_layer(conditional_enc(input_tensor, goal_tensor))
        error = torch.mean((prediction - target) ** 2, dim=1)
        error = torch.mean(error) / 2

        print(error.item())
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    assert error.item() < 0.02

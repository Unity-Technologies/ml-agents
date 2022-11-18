from mlagents.torch_utils import torch
from unittest import mock
import pytest

from mlagents.trainers.torch_entities.encoders import (
    VectorInput,
    Normalizer,
    SmallVisualEncoder,
    FullyConnectedVisualEncoder,
    SimpleVisualEncoder,
    ResNetVisualEncoder,
    NatureVisualEncoder,
)


# This test will also reveal issues with states not being saved in the state_dict.
def compare_models(module_1, module_2):
    is_same = True
    for key_item_1, key_item_2 in zip(
        module_1.state_dict().items(), module_2.state_dict().items()
    ):
        # Compare tensors in state_dict and not the keys.
        is_same = torch.equal(key_item_1[1], key_item_2[1]) and is_same
    return is_same


def test_normalizer():
    input_size = 2
    norm = Normalizer(input_size)

    # These three inputs should mean to 0.5, and variance 2
    # with the steps starting at 1
    vec_input1 = torch.tensor([[1, 1]])
    vec_input2 = torch.tensor([[1, 1]])
    vec_input3 = torch.tensor([[0, 0]])
    norm.update(vec_input1)
    norm.update(vec_input2)
    norm.update(vec_input3)

    # Test normalization
    for val in norm(vec_input1)[0].tolist():
        assert val == pytest.approx(0.707, abs=0.001)

    # Test copy normalization
    norm2 = Normalizer(input_size)
    assert not compare_models(norm, norm2)
    norm2.copy_from(norm)
    assert compare_models(norm, norm2)
    for val in norm2(vec_input1)[0].tolist():
        assert val == pytest.approx(0.707, abs=0.001)


@mock.patch("mlagents.trainers.torch_entities.encoders.Normalizer")
def test_vector_encoder(mock_normalizer):
    mock_normalizer_inst = mock.Mock()
    mock_normalizer.return_value = mock_normalizer_inst
    input_size = 64
    normalize = False
    vector_encoder = VectorInput(input_size, normalize)
    output = vector_encoder(torch.ones((1, input_size)))
    assert output.shape == (1, input_size)

    normalize = True
    vector_encoder = VectorInput(input_size, normalize)
    new_vec = torch.ones((1, input_size))
    vector_encoder.update_normalization(new_vec)

    mock_normalizer.assert_called_with(input_size)
    mock_normalizer_inst.update.assert_called_with(new_vec)

    vector_encoder2 = VectorInput(input_size, normalize)
    vector_encoder.copy_normalization(vector_encoder2)
    mock_normalizer_inst.copy_from.assert_called_with(mock_normalizer_inst)


@pytest.mark.parametrize("image_size", [(36, 36, 3), (84, 84, 4), (256, 256, 5)])
@pytest.mark.parametrize(
    "vis_class",
    [
        SimpleVisualEncoder,
        ResNetVisualEncoder,
        NatureVisualEncoder,
        SmallVisualEncoder,
        FullyConnectedVisualEncoder,
    ],
)
def test_visual_encoder(vis_class, image_size):
    num_outputs = 128
    enc = vis_class(image_size[0], image_size[1], image_size[2], num_outputs)
    # Note: NCHW not NHWC
    sample_input = torch.ones((1, image_size[0], image_size[1], image_size[2]))
    encoding = enc(sample_input)
    assert encoding.shape == (1, num_outputs)


@pytest.mark.parametrize(
    "vis_class, size",
    [
        (SimpleVisualEncoder, 36),
        (ResNetVisualEncoder, 36),
        (NatureVisualEncoder, 36),
        (SmallVisualEncoder, 10),
        (FullyConnectedVisualEncoder, 36),
    ],
)
@pytest.mark.slow
def test_visual_encoder_trains(vis_class, size):
    torch.manual_seed(0)
    image_size = (size, size, 1)
    batch = 100

    inputs = torch.cat(
        [torch.zeros((batch,) + image_size), torch.ones((batch,) + image_size)], dim=0
    )
    target = torch.cat([torch.zeros((batch,)), torch.ones((batch,))], dim=0)
    enc = vis_class(image_size[0], image_size[1], image_size[2], 1)
    optimizer = torch.optim.Adam(enc.parameters(), lr=0.001)

    for _ in range(15):
        prediction = enc(inputs)[:, 0]
        loss = torch.mean((target - prediction) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05

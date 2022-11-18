import pytest
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.settings import EncoderType, ScheduleType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes


def test_min_visual_size():
    # Make sure each EncoderType has an entry in MIS_RESOLUTION_FOR_ENCODER
    assert set(ModelUtils.MIN_RESOLUTION_FOR_ENCODER.keys()) == set(EncoderType)

    for encoder_type in EncoderType:
        good_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type]
        vis_input = torch.ones((1, good_size, good_size, 3))
        ModelUtils._check_resolution_for_encoder(good_size, good_size, encoder_type)
        enc_func = ModelUtils.get_encoder_for_type(encoder_type)
        enc = enc_func(good_size, good_size, 3, 1)
        enc.forward(vis_input)

        # Anything under the min size should raise an exception. If not, decrease the min size!
        with pytest.raises(Exception):
            bad_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type] - 1
            vis_input = torch.ones((1, bad_size, bad_size, 3))

            with pytest.raises(UnityTrainerException):
                # Make sure we'd hit a friendly error during model setup time.
                ModelUtils._check_resolution_for_encoder(
                    bad_size, bad_size, encoder_type
                )

            enc = enc_func(bad_size, bad_size, 3, 1)
            enc.forward(vis_input)


@pytest.mark.parametrize(
    "encoder_type",
    [
        EncoderType.SIMPLE,
        EncoderType.NATURE_CNN,
        EncoderType.SIMPLE,
        EncoderType.MATCH3,
    ],
)
def test_invalid_visual_input_size(encoder_type):
    with pytest.raises(UnityTrainerException):
        obs_spec = create_observation_specs_with_shapes(
            [
                (
                    ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type] - 1,
                    ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type],
                    1,
                )
            ]
        )
        ModelUtils.create_input_processors(obs_spec, 20, encoder_type, 20, False)


@pytest.mark.parametrize("num_visual", [0, 1, 2])
@pytest.mark.parametrize("num_vector", [0, 1, 2])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("encoder_type", [EncoderType.SIMPLE, EncoderType.NATURE_CNN])
def test_create_inputs(encoder_type, normalize, num_vector, num_visual):
    vec_obs_shape = (5,)
    vis_obs_shape = (84, 84, 3)
    obs_shapes = []
    for _ in range(num_vector):
        obs_shapes.append(vec_obs_shape)
    for _ in range(num_visual):
        obs_shapes.append(vis_obs_shape)
    h_size = 128
    obs_spec = create_observation_specs_with_shapes(obs_shapes)
    encoders, embedding_sizes = ModelUtils.create_input_processors(
        obs_spec, h_size, encoder_type, h_size, normalize
    )
    total_output = sum(embedding_sizes)
    vec_enc = []
    vis_enc = []
    for i, enc in enumerate(encoders):
        if len(obs_shapes[i]) == 1:
            vec_enc.append(enc)
        else:
            vis_enc.append(enc)
    assert len(vec_enc) == num_vector
    assert len(vis_enc) == num_visual
    assert total_output == int(num_visual * h_size + vec_obs_shape[0] * num_vector)
    if num_vector > 0:
        assert isinstance(vec_enc[0], VectorInput)

    for enc in vis_enc:
        assert isinstance(enc, ModelUtils.get_encoder_for_type(encoder_type))


def test_decayed_value():
    test_steps = [0, 4, 9]
    # Test constant decay
    param = ModelUtils.DecayedValue(ScheduleType.CONSTANT, 1.0, 0.2, test_steps[-1])
    for _step in test_steps:
        _param = param.get_value(_step)
        assert _param == 1.0

    test_results = [1.0, 0.6444, 0.2]
    # Test linear decay
    param = ModelUtils.DecayedValue(ScheduleType.LINEAR, 1.0, 0.2, test_steps[-1])
    for _step, _result in zip(test_steps, test_results):
        _param = param.get_value(_step)
        assert _param == pytest.approx(_result, abs=0.01)

    # Test invalid
    with pytest.raises(UnityTrainerException):
        ModelUtils.DecayedValue(
            "SomeOtherSchedule", 1.0, 0.2, test_steps[-1]
        ).get_value(0)


def test_polynomial_decay():
    test_steps = [0, 4, 9]
    test_results = [1.0, 0.7, 0.2]
    for _step, _result in zip(test_steps, test_results):
        decayed = ModelUtils.polynomial_decay(
            1.0, 0.2, test_steps[-1], _step, power=0.8
        )
        assert decayed == pytest.approx(_result, abs=0.01)


def test_list_to_tensor():
    # Test converting pure list
    unconverted_list = [[1.0, 2], [1, 3], [1, 4]]
    tensor = ModelUtils.list_to_tensor(unconverted_list)
    # Should be equivalent to torch.tensor conversion
    assert torch.equal(tensor, torch.tensor(unconverted_list))

    # Test converting pure numpy array
    np_list = np.asarray(unconverted_list)
    tensor = ModelUtils.list_to_tensor(np_list)
    # Should be equivalent to torch.tensor conversion
    assert torch.equal(tensor, torch.tensor(unconverted_list))

    # Test converting list of numpy arrays
    list_of_np = [np.asarray(_el) for _el in unconverted_list]
    tensor = ModelUtils.list_to_tensor(list_of_np)
    # Should be equivalent to torch.tensor conversion
    assert torch.equal(tensor, torch.tensor(unconverted_list, dtype=torch.float32))


def test_break_into_branches():
    # Test normal multi-branch case
    all_actions = torch.tensor([[1, 2, 3, 4, 5, 6]])
    action_size = [2, 1, 3]
    broken_actions = ModelUtils.break_into_branches(all_actions, action_size)
    assert len(action_size) == len(broken_actions)
    for i, _action in enumerate(broken_actions):
        assert _action.shape == (1, action_size[i])

    # Test 1-branch case
    action_size = [6]
    broken_actions = ModelUtils.break_into_branches(all_actions, action_size)
    assert len(broken_actions) == 1
    assert broken_actions[0].shape == (1, 6)


def test_actions_to_onehot():
    all_actions = torch.tensor([[1, 0, 2], [1, 0, 2]])
    action_size = [2, 1, 3]
    oh_actions = ModelUtils.actions_to_onehot(all_actions, action_size)
    expected_result = [
        torch.tensor([[0, 1], [0, 1]], dtype=torch.float),
        torch.tensor([[1], [1]], dtype=torch.float),
        torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float),
    ]
    for res, exp in zip(oh_actions, expected_result):
        assert torch.equal(res, exp)


def test_masked_mean():
    test_input = torch.tensor([1, 2, 3, 4, 5])
    masks = torch.ones_like(test_input).bool()
    mean = ModelUtils.masked_mean(test_input, masks=masks)
    assert mean == 3.0

    masks = torch.tensor([False, False, True, True, True])
    mean = ModelUtils.masked_mean(test_input, masks=masks)
    assert mean == 4.0

    # Make sure it works if all masks are off
    masks = torch.tensor([False, False, False, False, False])
    mean = ModelUtils.masked_mean(test_input, masks=masks)
    assert mean == 0.0

    # Make sure it works with 2d arrays of shape (mask_length, N)
    test_input = torch.tensor([1, 2, 3, 4, 5]).repeat(2, 1).T
    masks = torch.tensor([False, False, True, True, True])
    mean = ModelUtils.masked_mean(test_input, masks=masks)
    assert mean == 4.0


def test_soft_update():
    class TestModule(torch.nn.Module):
        def __init__(self, vals):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.ones(5, 5, 5) * vals)

    tm1 = TestModule(0)
    tm2 = TestModule(1)
    tm3 = TestModule(2)

    ModelUtils.soft_update(tm1, tm3, tau=0.5)
    assert torch.equal(tm3.parameter, torch.ones(5, 5, 5))

    ModelUtils.soft_update(tm1, tm2, tau=1.0)
    assert torch.equal(tm2.parameter, tm1.parameter)

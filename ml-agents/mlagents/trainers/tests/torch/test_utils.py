import pytest
import torch

from mlagents.trainers.settings import EncoderType
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.torch.encoders import (
    VectorEncoder,
    VectorAndUnnormalizedInputEncoder,
)


def test_min_visual_size():
    # Make sure each EncoderType has an entry in MIS_RESOLUTION_FOR_ENCODER
    assert set(ModelUtils.MIN_RESOLUTION_FOR_ENCODER.keys()) == set(EncoderType)

    for encoder_type in EncoderType:
        good_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type]
        vis_input = torch.ones((1, 3, good_size, good_size))
        ModelUtils._check_resolution_for_encoder(vis_input, encoder_type)
        enc_func = ModelUtils.get_encoder_for_type(encoder_type)
        enc = enc_func(good_size, good_size, 3, 1)
        enc.forward(vis_input)

        # Anything under the min size should raise an exception. If not, decrease the min size!
        with pytest.raises(Exception):
            bad_size = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[encoder_type] - 1
            vis_input = torch.ones((1, 3, bad_size, bad_size))

            with pytest.raises(UnityTrainerException):
                # Make sure we'd hit a friendly error during model setup time.
                ModelUtils._check_resolution_for_encoder(vis_input, encoder_type)

            enc = enc_func(bad_size, bad_size, 3, 1)
            enc.forward(vis_input)


@pytest.mark.parametrize("unnormalized_inputs", [0, 1])
@pytest.mark.parametrize("num_visual", [0, 1, 2])
@pytest.mark.parametrize("num_vector", [0, 1, 2])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("encoder_type", [EncoderType.SIMPLE, EncoderType.NATURE_CNN])
def test_create_encoders(
    encoder_type, normalize, num_vector, num_visual, unnormalized_inputs
):
    vec_obs_shape = (5,)
    vis_obs_shape = (84, 84, 3)
    obs_shapes = []
    for _ in range(num_vector):
        obs_shapes.append(vec_obs_shape)
    for _ in range(num_visual):
        obs_shapes.append(vis_obs_shape)
    h_size = 128
    num_layers = 3
    unnormalized_inputs = 1
    vis_enc, vec_enc = ModelUtils.create_encoders(
        obs_shapes, h_size, num_layers, encoder_type, unnormalized_inputs, normalize
    )
    vec_enc = list(vec_enc)
    vis_enc = list(vis_enc)
    assert len(vec_enc) == (
        1 if unnormalized_inputs + num_vector > 0 else 0
    )  # There's always at most one vector encoder.
    assert len(vis_enc) == num_visual

    if unnormalized_inputs > 0:
        assert isinstance(vec_enc[0], VectorAndUnnormalizedInputEncoder)
    elif num_vector > 0:
        assert isinstance(vec_enc[0], VectorEncoder)

    for enc in vis_enc:
        assert isinstance(enc, ModelUtils.get_encoder_for_type(encoder_type))

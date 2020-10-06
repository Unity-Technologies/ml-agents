import pytest

from mlagents.trainers.tf.models import ModelUtils
from mlagents.tf_utils import tf
from mlagents_envs.base_env import BehaviorSpec, ActionType


def create_behavior_spec(num_visual, num_vector, vector_size):
    behavior_spec = BehaviorSpec(
        [(84, 84, 3)] * int(num_visual) + [(vector_size,)] * int(num_vector),
        ActionType.DISCRETE,
        (1,),
    )
    return behavior_spec


@pytest.mark.parametrize("num_visual", [1, 2, 4])
@pytest.mark.parametrize("num_vector", [1, 2, 4])
def test_create_input_placeholders(num_vector, num_visual):
    vec_size = 8
    name_prefix = "test123"
    bspec = create_behavior_spec(num_visual, num_vector, vec_size)
    vec_in, vis_in = ModelUtils.create_input_placeholders(
        bspec.observation_shapes, name_prefix=name_prefix
    )

    assert isinstance(vis_in, list)
    assert len(vis_in) == num_visual
    assert isinstance(vec_in, tf.Tensor)
    assert vec_in.get_shape().as_list()[1] == num_vector * 8

    # Check names contain prefix and vis shapes are correct
    for _vis in vis_in:
        assert _vis.get_shape().as_list() == [None, 84, 84, 3]
        assert _vis.name.startswith(name_prefix)
    assert vec_in.name.startswith(name_prefix)

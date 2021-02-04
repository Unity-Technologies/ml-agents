from mlagents.trainers.tests.mock_brain import make_fake_trajectory
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix

VEC_OBS_SIZE = 6
ACTION_SIZE = 4


def test_trajectory_to_agentbuffer():
    length = 15
    wanted_keys = [
        (ObservationKeyPrefix.OBSERVATION, 0),
        (ObservationKeyPrefix.OBSERVATION, 1),
        (ObservationKeyPrefix.NEXT_OBSERVATION, 0),
        (ObservationKeyPrefix.NEXT_OBSERVATION, 1),
        BufferKey.MEMORY,
        BufferKey.MASKS,
        BufferKey.DONE,
        BufferKey.CONTINUOUS_ACTION,
        BufferKey.DISCRETE_ACTION,
        BufferKey.CONTINUOUS_LOG_PROBS,
        BufferKey.DISCRETE_LOG_PROBS,
        BufferKey.ACTION_MASK,
        BufferKey.PREV_ACTION,
        BufferKey.ENVIRONMENT_REWARDS,
    ]
    wanted_keys = set(wanted_keys)
    trajectory = make_fake_trajectory(
        length=length,
        observation_specs=create_observation_specs_with_shapes(
            [(VEC_OBS_SIZE,), (84, 84, 3)]
        ),
        action_spec=ActionSpec.create_continuous(ACTION_SIZE),
    )
    agentbuffer = trajectory.to_agentbuffer()
    seen_keys = set()
    for key, field in agentbuffer.items():
        assert len(field) == length
        seen_keys.add(key)

    assert seen_keys == wanted_keys

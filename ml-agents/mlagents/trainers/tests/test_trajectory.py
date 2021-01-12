from mlagents.trainers.tests.mock_brain import make_fake_trajectory
from mlagents.trainers.tests.dummy_config import create_sensor_specs_with_shapes
from mlagents_envs.base_env import ActionSpec

VEC_OBS_SIZE = 6
ACTION_SIZE = 4


def test_trajectory_to_agentbuffer():
    length = 15
    wanted_keys = [
        "next_obs_0",
        "next_obs_1",
        "obs_0",
        "obs_1",
        "memory",
        "masks",
        "done",
        "continuous_action",
        "discrete_action",
        "continuous_log_probs",
        "discrete_log_probs",
        "action_mask",
        "prev_action",
        "environment_rewards",
    ]
    wanted_keys = set(wanted_keys)
    trajectory = make_fake_trajectory(
        length=length,
        sensor_specs=create_sensor_specs_with_shapes([(VEC_OBS_SIZE,), (84, 84, 3)]),
        action_spec=ActionSpec.create_continuous(ACTION_SIZE),
    )
    agentbuffer = trajectory.to_agentbuffer()
    seen_keys = set()
    for key, field in agentbuffer.items():
        assert len(field) == length
        seen_keys.add(key)

    assert seen_keys == wanted_keys

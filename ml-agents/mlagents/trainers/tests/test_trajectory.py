import numpy as np
import pytest

from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.tests.mock_brain import make_fake_trajectory

from mlagents_envs.base_env import ActionSpec

VEC_OBS_SIZE = 6
ACTION_SIZE = 4


@pytest.mark.parametrize("num_visual_obs", [0, 1, 2])
@pytest.mark.parametrize("num_vec_obs", [0, 1])
def test_split_obs(num_visual_obs, num_vec_obs):
    obs = []
    for _ in range(num_visual_obs):
        obs.append(np.ones((84, 84, 3), dtype=np.float32))
    for _ in range(num_vec_obs):
        obs.append(np.ones(VEC_OBS_SIZE, dtype=np.float32))
    split_observations = SplitObservations.from_observations(obs)

    if num_vec_obs == 1:
        assert len(split_observations.vector_observations) == VEC_OBS_SIZE
    else:
        assert len(split_observations.vector_observations) == 0

    # Assert the number of vector observations.
    assert len(split_observations.visual_observations) == num_visual_obs


def test_trajectory_to_agentbuffer():
    length = 15
    wanted_keys = [
        "next_visual_obs0",
        "visual_obs0",
        "vector_obs",
        "next_vector_in",
        "memory",
        "masks",
        "done",
        "actions_pre",
        "actions",
        "action_probs",
        "action_mask",
        "prev_action",
        "environment_rewards",
    ]
    wanted_keys = set(wanted_keys)
    trajectory = make_fake_trajectory(
        length=length,
        observation_shapes=[(VEC_OBS_SIZE,), (84, 84, 3)],
        action_spec=ActionSpec.create_continuous(ACTION_SIZE),
    )
    agentbuffer = trajectory.to_agentbuffer()
    seen_keys = set()
    for key, field in agentbuffer.items():
        assert len(field) == length
        seen_keys.add(key)

    assert seen_keys == wanted_keys

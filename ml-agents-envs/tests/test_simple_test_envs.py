import numpy as np
import pytest

from mlagents_envs.base_env import ActionTuple
from simple_test_envs import STEP_SIZE, SimpleEnvironment


@pytest.mark.parametrize(
    "action_sizes,action",
    [
        (
            (1, 0),
            ActionTuple(continuous=np.array([[1.0]], dtype=np.float32)),
        ),
        (
            (0, 1),
            ActionTuple(discrete=np.array([[1]], dtype=np.int32)),
        ),
    ],
    ids=["continuous", "discrete"],
)
def test_simple_environment_applies_actions(action_sizes, action):
    env = SimpleEnvironment(["test"], action_sizes=action_sizes)
    env.reset()
    env.set_actions("test", action)

    env.step()

    np.testing.assert_allclose(env.positions["test"], [STEP_SIZE])
    assert env.step_count["test"] == 1

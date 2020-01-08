from unittest import mock
import pytest

import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import BatchedStepResult
from mlagents_envs.exception import UnityEnvironmentException, UnityActionException
from mlagents_envs.mock_communicator import MockCommunicator


@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_handles_bad_filename(get_communicator):
    with pytest.raises(UnityEnvironmentException):
        UnityEnvironment(" ")


@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_initialization(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    assert env.get_agent_groups() == ["RealFakeBrain"]
    env.close()


@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_reset(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    spec = env.get_agent_group_spec("RealFakeBrain")
    env.reset()
    batched_step_result = env.get_step_result("RealFakeBrain")
    env.close()
    assert isinstance(batched_step_result, BatchedStepResult)
    assert len(spec.observation_shapes) == len(batched_step_result.obs)
    n_agents = batched_step_result.n_agents()
    for shape, obs in zip(spec.observation_shapes, batched_step_result.obs):
        assert (n_agents,) + shape == obs.shape


@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_step(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    spec = env.get_agent_group_spec("RealFakeBrain")
    env.step()
    batched_step_result = env.get_step_result("RealFakeBrain")
    n_agents = batched_step_result.n_agents()
    env.set_actions(
        "RealFakeBrain", np.zeros((n_agents, spec.action_size), dtype=np.float32)
    )
    env.step()
    with pytest.raises(UnityActionException):
        env.set_actions(
            "RealFakeBrain",
            np.zeros((n_agents - 1, spec.action_size), dtype=np.float32),
        )
    batched_step_result = env.get_step_result("RealFakeBrain")
    n_agents = batched_step_result.n_agents()
    env.set_actions(
        "RealFakeBrain", -1 * np.ones((n_agents, spec.action_size), dtype=np.float32)
    )
    env.step()

    env.close()
    assert isinstance(batched_step_result, BatchedStepResult)
    assert len(spec.observation_shapes) == len(batched_step_result.obs)
    for shape, obs in zip(spec.observation_shapes, batched_step_result.obs):
        assert (n_agents,) + shape == obs.shape
    assert not batched_step_result.done[0]
    assert batched_step_result.done[2]


@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_close(mock_communicator, mock_launcher):
    comm = MockCommunicator(discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = comm
    env = UnityEnvironment(" ")
    assert env._loaded
    env.close()
    assert not env._loaded
    assert comm.has_been_closed


def test_returncode_to_signal_name():
    assert UnityEnvironment.returncode_to_signal_name(-2) == "SIGINT"
    assert UnityEnvironment.returncode_to_signal_name(42) is None
    assert UnityEnvironment.returncode_to_signal_name("SIGINT") is None


if __name__ == "__main__":
    pytest.main()

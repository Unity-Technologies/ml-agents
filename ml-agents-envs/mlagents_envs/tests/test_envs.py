from unittest import mock
import pytest

import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
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
    assert env.get_behavior_names() == ["RealFakeBrain"]
    env.close()


@pytest.mark.parametrize(
    "base_port,file_name,expected",
    [
        # Non-None base port value will always be used
        (6001, "foo.exe", 6001),
        # No port specified and environment specified, so use BASE_ENVIRONMENT_PORT
        (None, "foo.exe", UnityEnvironment.BASE_ENVIRONMENT_PORT),
        # No port specified and no environment, so use DEFAULT_EDITOR_PORT
        (None, None, UnityEnvironment.DEFAULT_EDITOR_PORT),
    ],
)
@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_port_defaults(
    mock_communicator, mock_launcher, base_port, file_name, expected
):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(file_name=file_name, worker_id=0, base_port=base_port)
    assert expected == env.port


@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_reset(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    spec = env.get_behavior_spec("RealFakeBrain")
    env.reset()
    decision_steps, terminal_steps = env.get_steps("RealFakeBrain")
    env.close()
    assert isinstance(decision_steps, DecisionSteps)
    assert isinstance(terminal_steps, TerminalSteps)
    assert len(spec.observation_shapes) == len(decision_steps.obs)
    assert len(spec.observation_shapes) == len(terminal_steps.obs)
    n_agents = len(decision_steps)
    for shape, obs in zip(spec.observation_shapes, decision_steps.obs):
        assert (n_agents,) + shape == obs.shape
    n_agents = len(terminal_steps)
    for shape, obs in zip(spec.observation_shapes, terminal_steps.obs):
        assert (n_agents,) + shape == obs.shape


@mock.patch("mlagents_envs.environment.UnityEnvironment.executable_launcher")
@mock.patch("mlagents_envs.environment.UnityEnvironment.get_communicator")
def test_step(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    spec = env.get_behavior_spec("RealFakeBrain")
    env.step()
    decision_steps, terminal_steps = env.get_steps("RealFakeBrain")
    n_agents = len(decision_steps)
    env.set_actions(
        "RealFakeBrain", np.zeros((n_agents, spec.action_size), dtype=np.float32)
    )
    env.step()
    with pytest.raises(UnityActionException):
        env.set_actions(
            "RealFakeBrain",
            np.zeros((n_agents - 1, spec.action_size), dtype=np.float32),
        )
    decision_steps, terminal_steps = env.get_steps("RealFakeBrain")
    n_agents = len(decision_steps)
    env.set_actions(
        "RealFakeBrain", -1 * np.ones((n_agents, spec.action_size), dtype=np.float32)
    )
    env.step()

    env.close()
    assert isinstance(decision_steps, DecisionSteps)
    assert isinstance(terminal_steps, TerminalSteps)
    assert len(spec.observation_shapes) == len(decision_steps.obs)
    assert len(spec.observation_shapes) == len(terminal_steps.obs)
    for shape, obs in zip(spec.observation_shapes, decision_steps.obs):
        assert (n_agents,) + shape == obs.shape
    assert 0 in decision_steps
    assert 2 in terminal_steps


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

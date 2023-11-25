import os
from unittest import mock
import pytest

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple
from mlagents_envs.exception import UnityEnvironmentException, UnityActionException
from mlagents_envs.mock_communicator import MockCommunicator


@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_handles_bad_filename(get_communicator):
    with pytest.raises(UnityEnvironmentException):
        UnityEnvironment(" ")


@mock.patch("mlagents_envs.env_utils.launch_executable")
@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_initialization(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    assert list(env.behavior_specs.keys()) == ["RealFakeBrain"]
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
@mock.patch("mlagents_envs.env_utils.launch_executable")
@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_port_defaults(
    mock_communicator, mock_launcher, base_port, file_name, expected
):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(file_name=file_name, worker_id=0, base_port=base_port)
    assert expected == env._port
    env.close()


@mock.patch("mlagents_envs.env_utils.launch_executable")
@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_log_file_path_is_set(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator()
    env = UnityEnvironment(
        file_name="myfile",
        worker_id=0,
        log_folder=os.path.join(".", "some-log-folder-path"),
    )
    args = env._executable_args()
    log_file_index = args.index("-logFile")
    assert args[log_file_index + 1] == os.path.join(
        ".", "some-log-folder-path", "Player-0.log"
    )
    env.close()


@mock.patch("mlagents_envs.env_utils.launch_executable")
@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_reset(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    spec = env.behavior_specs["RealFakeBrain"]
    env.reset()
    decision_steps, terminal_steps = env.get_steps("RealFakeBrain")
    env.close()
    assert isinstance(decision_steps, DecisionSteps)
    assert isinstance(terminal_steps, TerminalSteps)
    assert len(spec.observation_specs) == len(decision_steps.obs)
    assert len(spec.observation_specs) == len(terminal_steps.obs)
    n_agents = len(decision_steps)
    for sen_spec, obs in zip(spec.observation_specs, decision_steps.obs):
        assert (n_agents,) + sen_spec.shape == obs.shape
    n_agents = len(terminal_steps)
    for sen_spec, obs in zip(spec.observation_specs, terminal_steps.obs):
        assert (n_agents,) + sen_spec.shape == obs.shape


@mock.patch("mlagents_envs.env_utils.launch_executable")
@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_step(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0
    )
    env = UnityEnvironment(" ")
    spec = env.behavior_specs["RealFakeBrain"]
    env.step()
    decision_steps, terminal_steps = env.get_steps("RealFakeBrain")
    n_agents = len(decision_steps)
    env.set_actions("RealFakeBrain", spec.action_spec.empty_action(n_agents))
    env.step()
    with pytest.raises(UnityActionException):
        env.set_actions("RealFakeBrain", spec.action_spec.empty_action(n_agents - 1))
    decision_steps, terminal_steps = env.get_steps("RealFakeBrain")
    n_agents = len(decision_steps)
    _empty_act = spec.action_spec.empty_action(n_agents)
    next_action = ActionTuple(_empty_act.continuous - 1, _empty_act.discrete - 1)
    env.set_actions("RealFakeBrain", next_action)
    env.step()

    env.close()
    assert isinstance(decision_steps, DecisionSteps)
    assert isinstance(terminal_steps, TerminalSteps)
    assert len(spec.observation_specs) == len(decision_steps.obs)
    assert len(spec.observation_specs) == len(terminal_steps.obs)
    for spec, obs in zip(spec.observation_specs, decision_steps.obs):
        assert (n_agents,) + spec.shape == obs.shape
    assert 0 in decision_steps
    assert 2 in terminal_steps


@mock.patch("mlagents_envs.env_utils.launch_executable")
@mock.patch("mlagents_envs.environment.UnityEnvironment._get_communicator")
def test_close(mock_communicator, mock_launcher):
    comm = MockCommunicator(discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = comm
    env = UnityEnvironment(" ")
    assert env._loaded
    env.close()
    assert not env._loaded
    assert comm.has_been_closed


def test_check_communication_compatibility():
    unity_ver = "1.0.0"
    python_ver = "1.0.0"
    unity_package_version = "0.15.0"
    assert UnityEnvironment._check_communication_compatibility(
        unity_ver, python_ver, unity_package_version
    )
    unity_ver = "1.1.0"
    assert UnityEnvironment._check_communication_compatibility(
        unity_ver, python_ver, unity_package_version
    )
    unity_ver = "2.0.0"
    assert not UnityEnvironment._check_communication_compatibility(
        unity_ver, python_ver, unity_package_version
    )

    unity_ver = "0.16.0"
    python_ver = "0.16.0"
    assert UnityEnvironment._check_communication_compatibility(
        unity_ver, python_ver, unity_package_version
    )
    unity_ver = "0.17.0"
    assert not UnityEnvironment._check_communication_compatibility(
        unity_ver, python_ver, unity_package_version
    )
    unity_ver = "1.16.0"
    assert not UnityEnvironment._check_communication_compatibility(
        unity_ver, python_ver, unity_package_version
    )


def test_returncode_to_signal_name():
    assert UnityEnvironment._returncode_to_signal_name(-2) == "SIGINT"
    assert UnityEnvironment._returncode_to_signal_name(42) is None
    assert UnityEnvironment._returncode_to_signal_name("SIGINT") is None


if __name__ == "__main__":
    pytest.main()

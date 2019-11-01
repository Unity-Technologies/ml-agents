import unittest.mock as mock
import pytest
import numpy as np

from gym import spaces
from gym_unity.envs import UnityEnv, UnityGymException
from mlagents.envs.brain import CameraResolution


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_gym_wrapper(mock_env):
    mock_brain = create_mock_brainparams()
    mock_braininfo = create_mock_vector_braininfo()
    setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)

    env = UnityEnv(" ", use_visual=False, multiagent=False)
    assert isinstance(env, UnityEnv)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert env.observation_space.contains(obs)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_multi_agent(mock_env):
    mock_brain = create_mock_brainparams()
    mock_braininfo = create_mock_vector_braininfo(num_agents=2)
    setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)

    with pytest.raises(UnityGymException):
        UnityEnv(" ", multiagent=False)

    env = UnityEnv(" ", use_visual=False, multiagent=True)
    assert isinstance(env.reset(), list)
    actions = [env.action_space.sample() for i in range(env.number_agents)]
    obs, rew, done, info = env.step(actions)
    assert isinstance(obs, list)
    assert isinstance(rew, list)
    assert isinstance(done, list)
    assert isinstance(info, dict)


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_branched_flatten(mock_env):
    mock_brain = create_mock_brainparams(
        vector_action_space_type="discrete", vector_action_space_size=[2, 2, 3]
    )
    mock_braininfo = create_mock_vector_braininfo(num_agents=1)
    setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)

    env = UnityEnv(" ", use_visual=False, multiagent=False, flatten_branched=True)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 12
    assert env._flattener.lookup_action(0) == [0, 0, 0]
    assert env._flattener.lookup_action(11) == [1, 1, 2]

    # Check that False produces a MultiDiscrete
    env = UnityEnv(" ", use_visual=False, multiagent=False, flatten_branched=False)
    assert isinstance(env.action_space, spaces.MultiDiscrete)


@pytest.mark.parametrize("use_uint8", [True, False], ids=["float", "uint8"])
@mock.patch("gym_unity.envs.UnityEnvironment")
def test_gym_wrapper_visual(mock_env, use_uint8):
    mock_brain = create_mock_brainparams(number_visual_observations=1)
    mock_braininfo = create_mock_vector_braininfo(number_visual_observations=1)
    setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)

    env = UnityEnv(" ", use_visual=True, multiagent=False, uint8_visual=use_uint8)
    assert isinstance(env, UnityEnv)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert env.observation_space.contains(obs)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# Helper methods


def create_mock_brainparams(
    number_visual_observations=0,
    num_stacked_vector_observations=1,
    vector_action_space_type="continuous",
    vector_observation_space_size=3,
    vector_action_space_size=None,
):
    """
    Creates a mock BrainParameters object with parameters.
    """
    # Avoid using mutable object as default param
    if vector_action_space_size is None:
        vector_action_space_size = [2]
    mock_brain = mock.Mock()
    mock_brain.return_value.number_visual_observations = number_visual_observations
    if number_visual_observations:
        mock_brain.return_value.camera_resolutions = [
            CameraResolution(width=8, height=8, num_channels=3)
            for _ in range(number_visual_observations)
        ]
    mock_brain.return_value.num_stacked_vector_observations = (
        num_stacked_vector_observations
    )
    mock_brain.return_value.vector_action_space_type = vector_action_space_type
    mock_brain.return_value.vector_observation_space_size = (
        vector_observation_space_size
    )
    mock_brain.return_value.vector_action_space_size = vector_action_space_size
    return mock_brain()


def create_mock_vector_braininfo(num_agents=1, number_visual_observations=0):
    """
    Creates a mock BrainInfo with vector observations. Imitates constant
    vector observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate in your BrainInfo values.
    """
    mock_braininfo = mock.Mock()
    mock_braininfo.return_value.vector_observations = np.array([num_agents * [1, 2, 3]])
    if number_visual_observations:
        mock_braininfo.return_value.visual_observations = [[np.zeros(shape=(8, 8, 3))]]
    mock_braininfo.return_value.rewards = num_agents * [1.0]
    mock_braininfo.return_value.local_done = num_agents * [False]
    mock_braininfo.return_value.text_observations = num_agents * [""]
    mock_braininfo.return_value.agents = range(0, num_agents)
    return mock_braininfo()


def setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo):
    """
    Takes a mock UnityEnvironment and adds the appropriate properties, defined by the mock
    BrainParameters and BrainInfo.

    :Mock mock_env: A mock UnityEnvironment, usually empty.
    :Mock mock_brain: A mock Brain object that specifies the params of this environment.
    :Mock mock_braininfo: A mock BrainInfo object that will be returned at each step and reset.
    """
    mock_env.return_value.academy_name = "MockAcademy"
    mock_env.return_value.brains = {"MockBrain": mock_brain}
    mock_env.return_value.external_brain_names = ["MockBrain"]
    mock_env.return_value.reset.return_value = {"MockBrain": mock_braininfo}
    mock_env.return_value.step.return_value = {"MockBrain": mock_braininfo}

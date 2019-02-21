import unittest.mock as mock
import pytest
import numpy as np

from gym import spaces
from gym_unity.envs import UnityEnv, UnityGymException
from mock_communicator import MockCommunicator

@mock.patch('gym_unity.envs.unity_env.UnityEnvironment')
def test_gym_wrapper(mock_env):
    mock_env.return_value.academy_name = 'MockAcademy'
    mock_brain = mock.Mock();

    # Create mock Brain
    mock_brain.return_value.number_visual_observations = 0
    mock_brain.return_value.num_stacked_vector_observations = 1
    mock_brain.return_value.vector_action_space_type = 'continuous'
    mock_brain.return_value.vector_observation_space_size = 3
    mock_brain.return_value.vector_action_space_size = [2]

    # Create mock BrainInfo
    mock_braininfo = mock.Mock()
    mock_braininfo.return_value.vector_observations = np.array([[1, 2, 3,]])
    mock_braininfo.return_value.rewards = [1.0]
    mock_braininfo.return_value.local_done = [False]
    mock_braininfo.return_value.text_observations = ['']
    mock_braininfo.return_value.agents = [0]

    mock_env.return_value.brains = {'MockBrain':mock_brain()}
    mock_env.return_value.external_brain_names = ['MockBrain']
    mock_env.return_value.reset.return_value = {'MockBrain':mock_braininfo()}
    mock_env.return_value.step.return_value = {'MockBrain':mock_braininfo()}

    env = UnityEnv(' ', use_visual=False, multiagent=False)
    assert isinstance(env, UnityEnv)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

@mock.patch('gym_unity.envs.unity_env.UnityEnvironment')
def test_multi_agent(mock_env):
    mock_env.return_value.academy_name = 'MockAcademy'
    mock_brain = mock.Mock();

    # Create mock Brain
    mock_brain.return_value.number_visual_observations = 0
    mock_brain.return_value.num_stacked_vector_observations = 1
    mock_brain.return_value.vector_action_space_type = 'continuous'
    mock_brain.return_value.vector_observation_space_size = 3
    mock_brain.return_value.vector_action_space_size = [2]

    # Create mock BrainInfo
    mock_braininfo = mock.Mock()
    mock_braininfo.return_value.vector_observations = np.array([[1, 2, 3,],[1, 2, 3]])
    mock_braininfo.return_value.rewards = [1.0, 1.0]
    mock_braininfo.return_value.local_done = [False, False]
    mock_braininfo.return_value.text_observations = ['', '']
    mock_braininfo.return_value.agents = [0, 1]

    mock_env.return_value.brains = {'MockBrain':mock_brain()}
    mock_env.return_value.external_brain_names = ['MockBrain']
    mock_env.return_value.reset.return_value = {'MockBrain':mock_braininfo()}
    mock_env.return_value.step.return_value = {'MockBrain':mock_braininfo()}

    # Test for incorrect number of agents.
    with pytest.raises(UnityGymException):
        UnityEnv(' ', multiagent=False)

    env = UnityEnv(' ', use_visual=False, multiagent=True)
    assert isinstance(env.reset(), list)
    actions = [env.action_space.sample() for i in range(env.number_agents)]
    obs, rew, done, info = env.step(actions)
    assert isinstance(obs, list)
    assert isinstance(rew, list)
    assert isinstance(done, list)
    assert isinstance(info, dict)

@mock.patch('gym_unity.envs.unity_env.UnityEnvironment')
def test_branched_flatten(mock_env):
    mock_env.return_value.academy_name = 'MockAcademy'

    # Create mock Brain
    mock_brain = mock.Mock();
    mock_brain.return_value.number_visual_observations = 0
    mock_brain.return_value.num_stacked_vector_observations = 1
    mock_brain.return_value.vector_action_space_type = 'discrete'
    mock_brain.return_value.vector_observation_space_size = 1
    # Unflattened action space
    mock_brain.return_value.vector_action_space_size = [2,2,3]

    mock_env.return_value.brains = {'MockBrain':mock_brain()}
    mock_env.return_value.external_brain_names = ['MockBrain']
    env = UnityEnv(' ', use_visual=False, multiagent=False, flatten_branched=True)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n==12
    assert env._flattener.lookup_action(0)==[0,0,0]
    assert env._flattener.lookup_action(11)==[1,1,2]

    # Check that False produces a MultiDiscrete
    env = UnityEnv(' ', use_visual=False, multiagent=False, flatten_branched=False)
    assert isinstance(env.action_space, spaces.MultiDiscrete)

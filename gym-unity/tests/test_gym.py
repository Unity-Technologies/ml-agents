import unittest.mock as mock
import pytest
import numpy as np

from gym import spaces
from gym_unity.envs import UnityEnv, UnityGymException
from mock_communicator import MockCommunicator

@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_gym_wrapper(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0, stack=False, num_agents=1)

    # Test for incorrect number of agents.
    with pytest.raises(UnityGymException):
        UnityEnv(' ', use_visual=False, multiagent=True)

    env = UnityEnv(' ', use_visual=False)
    assert isinstance(env, UnityEnv)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_multi_agent(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0, stack=False, num_agents=2)

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

import unittest.mock as mock
import pytest
import numpy as np

from gym_unity.envs import UnityEnv, UnityGymException
from tests.mock_communicator import MockCommunicator

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

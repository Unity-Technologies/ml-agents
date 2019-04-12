import unittest.mock as mock
import pytest

import numpy as np

from mlagents.envs import UnityEnvironment, UnityEnvironmentException, UnityActionException, \
    BrainInfo
from mlagents.envs.mock_communicator import MockCommunicator


@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_handles_bad_filename(get_communicator):
    with pytest.raises(UnityEnvironmentException):
        UnityEnvironment(' ')


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_initialization(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    env = UnityEnvironment(' ')
    with pytest.raises(UnityActionException):
        env.step([0])
    assert env.brain_names[0] == 'RealFakeBrain'
    env.close()


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_reset(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    env = UnityEnvironment(' ')
    brain = env.brains['RealFakeBrain']
    brain_info = env.reset()
    env.close()
    assert not env.global_done
    assert isinstance(brain_info, dict)
    assert isinstance(brain_info['RealFakeBrain'], BrainInfo)
    assert isinstance(brain_info['RealFakeBrain'].visual_observations, list)
    assert isinstance(brain_info['RealFakeBrain'].vector_observations, np.ndarray)
    assert len(brain_info['RealFakeBrain'].visual_observations) == brain.number_visual_observations
    assert len(brain_info['RealFakeBrain'].vector_observations) == \
           len(brain_info['RealFakeBrain'].agents)
    assert len(brain_info['RealFakeBrain'].vector_observations[0]) == \
           brain.vector_observation_space_size * brain.num_stacked_vector_observations


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_step(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    env = UnityEnvironment(' ')
    brain = env.brains['RealFakeBrain']
    brain_info = env.reset()
    brain_info = env.step([0] * brain.vector_action_space_size[0] * len(brain_info['RealFakeBrain'].agents))
    with pytest.raises(UnityActionException):
        env.step([0])
    brain_info = env.step([-1] * brain.vector_action_space_size[0] * len(brain_info['RealFakeBrain'].agents))
    with pytest.raises(UnityActionException):
        env.step([0] * brain.vector_action_space_size[0] * len(brain_info['RealFakeBrain'].agents))
    env.close()
    assert env.global_done
    assert isinstance(brain_info, dict)
    assert isinstance(brain_info['RealFakeBrain'], BrainInfo)
    assert isinstance(brain_info['RealFakeBrain'].visual_observations, list)
    assert isinstance(brain_info['RealFakeBrain'].vector_observations, np.ndarray)
    assert len(brain_info['RealFakeBrain'].visual_observations) == brain.number_visual_observations
    assert len(brain_info['RealFakeBrain'].vector_observations) == \
           len(brain_info['RealFakeBrain'].agents)
    assert len(brain_info['RealFakeBrain'].vector_observations[0]) == \
           brain.vector_observation_space_size * brain.num_stacked_vector_observations

    print("\n\n\n\n\n\n\n" + str(brain_info['RealFakeBrain'].local_done))
    assert not brain_info['RealFakeBrain'].local_done[0]
    assert brain_info['RealFakeBrain'].local_done[2]


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_close(mock_communicator, mock_launcher):
    comm = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = comm
    env = UnityEnvironment(' ')
    assert env._loaded
    env.close()
    assert not env._loaded
    assert comm.has_been_closed


if __name__ == '__main__':
    pytest.main()

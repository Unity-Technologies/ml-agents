import json
import unittest.mock as mock
import pytest
import struct

import numpy as np

from unityagents import UnityEnvironment, UnityEnvironmentException, UnityActionException, \
    BrainInfo, Curriculum
from .mock_communicator import MockCommunicator


dummy_curriculum = json.loads('''{
    "measure" : "reward",
    "thresholds" : [10, 20, 50],
    "min_lesson_length" : 3,
    "signal_smoothing" : true, 
    "parameters" : 
    {
        "param1" : [0.7, 0.5, 0.3, 0.1],
        "param2" : [100, 50, 20, 15],
        "param3" : [0.2, 0.3, 0.7, 0.9]
    }
}''')
bad_curriculum = json.loads('''{
    "measure" : "reward",
    "thresholds" : [10, 20, 50],
    "min_lesson_length" : 3,
    "signal_smoothing" : false, 
    "parameters" : 
    {
        "param1" : [0.7, 0.5, 0.3, 0.1],
        "param2" : [100, 50, 20],
        "param3" : [0.2, 0.3, 0.7, 0.9]
    }
}''')


def test_handles_bad_filename():
    with pytest.raises(UnityEnvironmentException):
        UnityEnvironment(' ')


@mock.patch('unityagents.UnityEnvironment.executable_launcher')
@mock.patch('unityagents.UnityEnvironment.get_communicator')
def test_initialization(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    env = UnityEnvironment(' ')
    with pytest.raises(UnityActionException):
        env.step([0])
    assert env.brain_names[0] == 'RealFakeBrain'
    env.close()


@mock.patch('unityagents.UnityEnvironment.executable_launcher')
@mock.patch('unityagents.UnityEnvironment.get_communicator')
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
    assert brain_info['RealFakeBrain'].vector_observations.shape[0] == \
           len(brain_info['RealFakeBrain'].agents)
    assert brain_info['RealFakeBrain'].vector_observations.shape[1] == \
           brain.vector_observation_space_size * brain.num_stacked_vector_observations


@mock.patch('unityagents.UnityEnvironment.executable_launcher')
@mock.patch('unityagents.UnityEnvironment.get_communicator')
def test_step(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    env = UnityEnvironment(' ')
    brain = env.brains['RealFakeBrain']
    brain_info = env.reset()
    brain_info = env.step([0] * brain.vector_action_space_size * len(brain_info['RealFakeBrain'].agents))
    with pytest.raises(UnityActionException):
        env.step([0])
    brain_info = env.step([-1] * brain.vector_action_space_size * len(brain_info['RealFakeBrain'].agents))
    with pytest.raises(UnityActionException):
        env.step([0] * brain.vector_action_space_size * len(brain_info['RealFakeBrain'].agents))
    env.close()
    assert env.global_done
    assert isinstance(brain_info, dict)
    assert isinstance(brain_info['RealFakeBrain'], BrainInfo)
    assert isinstance(brain_info['RealFakeBrain'].visual_observations, list)
    assert isinstance(brain_info['RealFakeBrain'].vector_observations, np.ndarray)
    assert len(brain_info['RealFakeBrain'].visual_observations) == brain.number_visual_observations
    assert brain_info['RealFakeBrain'].vector_observations.shape[0] == \
           len(brain_info['RealFakeBrain'].agents)
    assert brain_info['RealFakeBrain'].vector_observations.shape[1] == \
           brain.vector_observation_space_size * brain.num_stacked_vector_observations

    print("\n\n\n\n\n\n\n" + str(brain_info['RealFakeBrain'].local_done))
    assert not brain_info['RealFakeBrain'].local_done[0]
    assert brain_info['RealFakeBrain'].local_done[2]


@mock.patch('unityagents.UnityEnvironment.executable_launcher')
@mock.patch('unityagents.UnityEnvironment.get_communicator')
def test_close(mock_communicator, mock_launcher):
    comm = MockCommunicator(
        discrete_action=False, visual_inputs=0)
    mock_communicator.return_value = comm
    env = UnityEnvironment(' ')
    assert env._loaded
    env.close()
    assert not env._loaded
    assert comm.has_been_closed


def test_curriculum():
    open_name = '%s.open' % __name__
    with mock.patch('json.load') as mock_load:
        with mock.patch(open_name, create=True) as mock_open:
            mock_open.return_value = 0
            mock_load.return_value = bad_curriculum
            with pytest.raises(UnityEnvironmentException):
                Curriculum('tests/test_unityagents.py', {"param1": 1, "param2": 1, "param3": 1})
            mock_load.return_value = dummy_curriculum
            with pytest.raises(UnityEnvironmentException):
                Curriculum('tests/test_unityagents.py', {"param1": 1, "param2": 1})
            curriculum = Curriculum('tests/test_unityagents.py', {"param1": 1, "param2": 1, "param3": 1})
            assert curriculum.get_lesson_number == 0
            curriculum.set_lesson_number(1)
            assert curriculum.get_lesson_number == 1
            curriculum.increment_lesson(10)
            assert curriculum.get_lesson_number == 1
            curriculum.increment_lesson(30)
            curriculum.increment_lesson(30)
            assert curriculum.get_lesson_number == 1
            assert curriculum.lesson_length == 3
            curriculum.increment_lesson(30)
            assert curriculum.get_config() == {'param1': 0.3, 'param2': 20, 'param3': 0.7}
            assert curriculum.get_config(0) == {"param1": 0.7, "param2": 100, "param3": 0.2}
            assert curriculum.lesson_length == 0
            assert curriculum.get_lesson_number == 2


if __name__ == '__main__':
    pytest.main()

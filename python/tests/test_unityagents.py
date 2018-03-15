import json
import unittest.mock as mock
import pytest
import struct

import numpy as np

from unityagents import UnityEnvironment, UnityEnvironmentException, UnityActionException, \
    BrainInfo, Curriculum


def append_length(partial_string):
    return struct.pack("I", len(partial_string.encode())) + partial_string.encode()


dummy_start = '''{
  "AcademyName": "RealFakeAcademy",
  "resetParameters": {},
  "brainNames": ["RealFakeBrain"],
  "externalBrainNames": ["RealFakeBrain"],
  "logPath":"RealFakePath",
  "apiNumber":"API-3",
  "brainParameters": [{
      "vectorObservationSize": 3,
      "numStackedVectorObservations": 2,
      "vectorActionSize": 2,
      "memorySize": 0,
      "cameraResolutions": [],
      "vectorActionDescriptions": ["",""],
      "vectorActionSpaceType": 1,
      "vectorObservationSpaceType": 1
      }]
}'''.encode()

dummy_reset = [
    'CONFIG_REQUEST'.encode(),
    append_length(
        '''
        {
          "brain_name": "RealFakeBrain",
          "agents": [1,2],
          "vectorObservations": [1,2,3,4,5,6,1,2,3,4,5,6],
          "rewards": [1,2],
          "previousVectorActions": [1,2,3,4],
          "previousTextActions":["",""],
          "memories": [],
          "dones": [false, false],
          "maxes": [false, false],
          "textObservations" :[" "," "]
        }'''),
    append_length('END_OF_MESSAGE:False')]

dummy_step = ['actions'.encode(),
              append_length('''
{
  "brain_name": "RealFakeBrain",
  "agents": [1,2,3],
  "vectorObservations": [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9],
  "rewards": [1,2,3],
  "previousVectorActions": [1,2,3,4,5,6],
  "previousTextActions":["","",""],
  "memories": [],
  "dones": [false, false, false],
  "maxes": [false, false, false],
  "textObservations" :[" "," ", " "]
}'''),
    append_length('END_OF_MESSAGE:False'),
              'actions'.encode(),
              append_length('''
{
  "brain_name": "RealFakeBrain",
  "agents": [1,2,3],
  "vectorObservations": [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9],
  "rewards": [1,2,3],
  "previousVectorActions": [1,2,3,4,5,6],
  "previousTextActions":["","",""],
  "memories": [],
  "dones": [false, false, true],
  "maxes": [false, false, false],
  "textObservations" :[" "," ", " "]
}'''),
    append_length('END_OF_MESSAGE:True')]

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


def test_initialization():
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['FakeLaunchPath']
                mock_socket.return_value.accept.return_value = (mock_socket, 0)
                mock_socket.recv.return_value.decode.return_value = dummy_start
                env = UnityEnvironment(' ')
                with pytest.raises(UnityActionException):
                    env.step([0])
                assert env.brain_names[0] == 'RealFakeBrain'
                env.close()


def test_reset():
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['FakeLaunchPath']
                mock_socket.return_value.accept.return_value = (mock_socket, 0)
                mock_socket.recv.return_value.decode.return_value = dummy_start
                env = UnityEnvironment(' ')
                brain = env.brains['RealFakeBrain']
                mock_socket.recv.side_effect = dummy_reset
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



def test_step():
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['FakeLaunchPath']
                mock_socket.return_value.accept.return_value = (mock_socket, 0)
                mock_socket.recv.return_value.decode.return_value = dummy_start
                env = UnityEnvironment(' ')
                brain = env.brains['RealFakeBrain']
                mock_socket.recv.side_effect = dummy_reset
                brain_info = env.reset()
                mock_socket.recv.side_effect = dummy_step
                brain_info = env.step([0] * brain.vector_action_space_size * len(brain_info['RealFakeBrain'].agents))
                with pytest.raises(UnityActionException):
                    env.step([0])
                brain_info = env.step([0] * brain.vector_action_space_size * len(brain_info['RealFakeBrain'].agents))
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
                assert not brain_info['RealFakeBrain'].local_done[0]
                assert brain_info['RealFakeBrain'].local_done[2]


def test_close():
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['FakeLaunchPath']
                mock_socket.return_value.accept.return_value = (mock_socket, 0)
                mock_socket.recv.return_value.decode.return_value = dummy_start
                env = UnityEnvironment(' ')
                assert env._loaded
                env.close()
                assert not env._loaded
                mock_socket.close.assert_called_once()


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

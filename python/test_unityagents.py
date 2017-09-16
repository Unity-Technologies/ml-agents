import mock
import numpy as np
import os
import pytest
import socket
import mock

from unityagents import UnityEnvironment, UnityEnvironmentException, UnityActionException, BrainInfo, BrainParameters

dummy_start = '''{
  "AcademyName": "RealFakeAcademy",
  "resetParameters": {},
  "brainNames": ["RealFakeBrain"],
  "brainParameters": [{
      "stateSize": 3,
      "actionSize": 2,
      "memorySize": 0,
      "cameraResolutions": [],
      "actionDescriptions": ["",""],
      "actionSpaceType": 1,
      "stateSpaceType": 1
      }]
}'''.encode()

dummy_reset = [
'CONFIG_REQUEST'.encode(),
'''
{
  "brain_name": "RealFakeBrain",
  "agents": [1,2],
  "states": [1,2,3,4,5,6],
  "rewards": [1,2],
  "actions": null,
  "memories": [],
  "dones": [false, false]
}'''.encode(),
'False'.encode()]

dummy_step = ['actions'.encode(),
'''
{
  "brain_name": "RealFakeBrain",
  "agents": [1,2,3],
  "states": [1,2,3,4,5,6,7,8,9],
  "rewards": [1,2,3],
  "actions": null,
  "memories": [],
  "dones": [false, false, false]
}'''.encode(),
'False'.encode(),
'actions'.encode(),
'''
{
  "brain_name": "RealFakeBrain",
  "agents": [1,2,3],
  "states": [1,2,3,4,5,6,7,8,9],
  "rewards": [1,2,3],
  "actions": null,
  "memories": [],
  "dones": [false, false, true]
}'''.encode(),
'True'.encode()]


def test_handles_bad_filename():
    with pytest.raises(UnityEnvironmentException):
        UnityEnvironment(' ')

def test_initialialization():
    with mock.patch('subprocess.Popen') as mock_subproc_popen:
        with mock.patch('socket.socket') as mock_socket:
            mock_socket.return_value.accept.return_value = (mock_socket, 0)
            mock_socket.recv.return_value.decode.return_value = dummy_start
            env = UnityEnvironment(' ')
            with pytest.raises(UnityActionException):
                env.step([0])
            assert env.brain_names[0] == 'RealFakeBrain'
            env.close()
            

def test_reset():
    with mock.patch('subprocess.Popen') as mock_subproc_popen:
        with mock.patch('socket.socket') as mock_socket:
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
            assert isinstance(brain_info['RealFakeBrain'].observations, list) 
            assert isinstance(brain_info['RealFakeBrain'].states, np.ndarray)
            assert len(brain_info['RealFakeBrain'].observations) == brain.number_observations
            assert brain_info['RealFakeBrain'].states.shape[0] == len(brain_info['RealFakeBrain'].agents)
            assert brain_info['RealFakeBrain'].states.shape[1] == brain.state_space_size 


def test_step():
    with mock.patch('subprocess.Popen') as mock_subproc_popen:
        with mock.patch('socket.socket') as mock_socket:
            mock_socket.return_value.accept.return_value = (mock_socket, 0)
            mock_socket.recv.return_value.decode.return_value = dummy_start
            env = UnityEnvironment(' ')
            brain = env.brains['RealFakeBrain']
            mock_socket.recv.side_effect = dummy_reset 
            brain_info = env.reset()
            mock_socket.recv.side_effect = dummy_step
            brain_info = env.step([0] * brain.action_space_size * len(brain_info['RealFakeBrain'].agents))
            with pytest.raises(UnityActionException):
                env.step([0])
            brain_info = env.step([0] * brain.action_space_size * len(brain_info['RealFakeBrain'].agents))
            with pytest.raises(UnityActionException):
                env.step([0] * brain.action_space_size * len(brain_info['RealFakeBrain'].agents))
            env.close()
            assert env.global_done
            assert isinstance(brain_info, dict)
            assert isinstance(brain_info['RealFakeBrain'], BrainInfo)
            assert isinstance(brain_info['RealFakeBrain'].observations, list) 
            assert isinstance(brain_info['RealFakeBrain'].states, np.ndarray)
            assert len(brain_info['RealFakeBrain'].observations) == brain.number_observations
            assert brain_info['RealFakeBrain'].states.shape[0] == len(brain_info['RealFakeBrain'].agents)
            assert brain_info['RealFakeBrain'].states.shape[1] == brain.state_space_size
            assert not brain_info['RealFakeBrain'].local_done[0]
            assert brain_info['RealFakeBrain'].local_done[2]




def test_close():
    with mock.patch('subprocess.Popen') as mock_subproc_popen:
        with mock.patch('socket.socket') as mock_socket:
            mock_socket.return_value.accept.return_value = (mock_socket, 0)
            mock_socket.recv.return_value.decode.return_value = dummy_start
            env = UnityEnvironment(' ')
            assert env._loaded
            env.close()
            assert not env._loaded
            mock_socket.close.assert_called_once()



if __name__ == '__main__':
    pytest.main()

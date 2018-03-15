import yaml
import unittest.mock as mock
import pytest

from unitytrainers.trainer_controller import TrainerController
from unitytrainers.buffer import Buffer
from unitytrainers.models import *
from unitytrainers.ppo.trainer import PPOTrainer
from unitytrainers.bc.trainer import BehavioralCloningTrainer
from unityagents import UnityEnvironmentException

dummy_start = '''{
  "AcademyName": "RealFakeAcademy",
  "resetParameters": {},
  "brainNames": ["RealFakeBrain"],
  "externalBrainNames": ["RealFakeBrain"],
  "logPath":"RealFakePath",
  "apiNumber":"API-3",
  "brainParameters": [{
      "vectorObservationSize": 3,
      "numStackedVectorObservations" : 2,
      "vectorActionSize": 2,
      "memorySize": 0,
      "cameraResolutions": [],
      "vectorActionDescriptions": ["",""],
      "vectorActionSpaceType": 1,
      "vectorObservationSpaceType": 1
      }]
}'''.encode()


dummy_config = yaml.load('''
default:
    trainer: ppo
    batch_size: 32
    beta: 5.0e-3
    buffer_size: 512
    epsilon: 0.2
    gamma: 0.99
    hidden_units: 128
    lambd: 0.95
    learning_rate: 3.0e-4
    max_steps: 5.0e4
    normalize: true
    num_epoch: 5
    num_layers: 2
    time_horizon: 64
    sequence_length: 64
    summary_freq: 1000
    use_recurrent: false
    memory_size: 8
''')

dummy_bc_config = yaml.load('''
default:
    trainer: imitation
    brain_to_imitate: ExpertBrain
    batches_per_epoch: 16
    batch_size: 32
    beta: 5.0e-3
    buffer_size: 512
    epsilon: 0.2
    gamma: 0.99
    hidden_units: 128
    lambd: 0.95
    learning_rate: 3.0e-4
    max_steps: 5.0e4
    normalize: true
    num_epoch: 5
    num_layers: 2
    time_horizon: 64
    sequence_length: 64
    summary_freq: 1000
    use_recurrent: false
    memory_size: 8
''')

dummy_bad_config = yaml.load('''
default:
    trainer: incorrect_trainer
    brain_to_imitate: ExpertBrain
    batches_per_epoch: 16
    batch_size: 32
    beta: 5.0e-3
    buffer_size: 512
    epsilon: 0.2
    gamma: 0.99
    hidden_units: 128
    lambd: 0.95
    learning_rate: 3.0e-4
    max_steps: 5.0e4
    normalize: true
    num_epoch: 5
    num_layers: 2
    time_horizon: 64
    sequence_length: 64
    summary_freq: 1000
    use_recurrent: false
    memory_size: 8
''')


def test_initialization():
    with mock.patch('subprocess.Popen'):
        with mock.patch('socket.socket') as mock_socket:
            with mock.patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['FakeLaunchPath']
                mock_socket.return_value.accept.return_value = (mock_socket, 0)
                mock_socket.recv.return_value.decode.return_value = dummy_start
                tc = TrainerController(' ', ' ', 1, None, True, True, False, 1,
                                       1, 1, 1, '', "tests/test_unitytrainers.py")
                assert(tc.env.brain_names[0] == 'RealFakeBrain')


def test_load_config():
    open_name = 'unitytrainers.trainer_controller' + '.open'
    with mock.patch('yaml.load') as mock_load:
        with mock.patch(open_name, create=True) as _:
            with mock.patch('subprocess.Popen'):
                with mock.patch('socket.socket') as mock_socket:
                    with mock.patch('glob.glob') as mock_glob:
                        mock_load.return_value = dummy_config
                        mock_glob.return_value = ['FakeLaunchPath']
                        mock_socket.return_value.accept.return_value = (mock_socket, 0)
                        mock_socket.recv.return_value.decode.return_value = dummy_start
                        mock_load.return_value = dummy_config
                        tc = TrainerController(' ', ' ', 1, None, True, True, False, 1,
                                                   1, 1, 1, '','')
                        config = tc._load_config()
                        assert(len(config) == 1)
                        assert(config['default']['trainer'] == "ppo")


def test_initialize_trainers():
    open_name = 'unitytrainers.trainer_controller' + '.open'
    with mock.patch('yaml.load') as mock_load:
        with mock.patch(open_name, create=True) as _:
            with mock.patch('subprocess.Popen'):
                with mock.patch('socket.socket') as mock_socket:
                    with mock.patch('glob.glob') as mock_glob:
                        mock_glob.return_value = ['FakeLaunchPath']
                        mock_socket.return_value.accept.return_value = (mock_socket, 0)
                        mock_socket.recv.return_value.decode.return_value = dummy_start
                        tc = TrainerController(' ', ' ', 1, None, True, True, False, 1,
                                               1, 1, 1, '', "tests/test_unitytrainers.py")

                        # Test for PPO trainer
                        mock_load.return_value = dummy_config
                        config = tc._load_config()
                        tf.reset_default_graph()
                        with tf.Session() as sess:
                            tc._initialize_trainers(config, sess)
                            assert(len(tc.trainers) == 1)
                            assert(isinstance(tc.trainers['RealFakeBrain'], PPOTrainer))

                        # Test for Behavior Cloning Trainer
                        mock_load.return_value = dummy_bc_config
                        config = tc._load_config()
                        tf.reset_default_graph()
                        with tf.Session() as sess:
                            tc._initialize_trainers(config, sess)
                            assert(isinstance(tc.trainers['RealFakeBrain'], BehavioralCloningTrainer))

                        # Test for proper exception when trainer name is incorrect
                        mock_load.return_value = dummy_bad_config
                        config = tc._load_config()
                        tf.reset_default_graph()
                        with tf.Session() as sess:
                            with pytest.raises(UnityEnvironmentException):
                                tc._initialize_trainers(config, sess)


def assert_array(a, b):
    assert a.shape == b.shape
    la = list(a.flatten())
    lb = list(b.flatten())
    for i in range(len(la)):
        assert la[i] == lb[i]


def test_buffer():
    b = Buffer()
    for fake_agent_id in range(4):
        for step in range(9):
            b[fake_agent_id]['vector_observation'].append(
                [100 * fake_agent_id + 10 * step + 1,
                 100 * fake_agent_id + 10 * step + 2,
                 100 * fake_agent_id + 10 * step + 3]
            )
            b[fake_agent_id]['action'].append([100 * fake_agent_id + 10 * step + 4,
                                               100 * fake_agent_id + 10 * step + 5])
    a = b[1]['vector_observation'].get_batch(batch_size=2, training_length=None, sequential=True)
    assert_array(a, np.array([[171, 172, 173], [181, 182, 183]]))
    a = b[2]['vector_observation'].get_batch(batch_size=2, training_length=3, sequential=True)
    assert_array(a, np.array([
        [[231, 232, 233], [241, 242, 243], [251, 252, 253]],
        [[261, 262, 263], [271, 272, 273], [281, 282, 283]]
    ]))
    a = b[2]['vector_observation'].get_batch(batch_size=2, training_length=3, sequential=False)
    assert_array(a, np.array([
        [[251, 252, 253], [261, 262, 263], [271, 272, 273]],
        [[261, 262, 263], [271, 272, 273], [281, 282, 283]]
    ]))
    b[4].reset_agent()
    assert len(b[4]) == 0
    b.append_update_buffer(3,
                           batch_size=None, training_length=2)
    b.append_update_buffer(2,
                           batch_size=None, training_length=2)
    assert len(b.update_buffer['action']) == 10
    assert np.array(b.update_buffer['action']).shape == (10, 2, 2)


if __name__ == '__main__':
    pytest.main()

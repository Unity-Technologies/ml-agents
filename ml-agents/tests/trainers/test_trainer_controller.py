import json
import unittest.mock as mock
import os

import yaml
import pytest
import tensorflow as tf

from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.bc.offline_trainer import OfflineBCTrainer
from mlagents.trainers.bc.online_trainer import OnlineBCTrainer
from mlagents.envs.exception import UnityEnvironmentException
from tests.mock_communicator import MockCommunicator


@pytest.fixture
def dummy_config():
    return yaml.load(
        '''
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
            use_curiosity: false
            curiosity_strength: 0.0
            curiosity_enc_size: 1
        ''')


@pytest.fixture
def dummy_online_bc_config():
    return yaml.load(
        '''
        default:
            trainer: online_bc
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
            use_curiosity: false
            curiosity_strength: 0.0
            curiosity_enc_size: 1
        ''')


@pytest.fixture
def dummy_offline_bc_config():
    return yaml.load(
        '''
        default:
            trainer: offline_bc
            demo_path: '''
        + os.path.dirname(os.path.abspath(__file__)) + '''/test.demo
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
            use_curiosity: false
            curiosity_strength: 0.0
            curiosity_enc_size: 1
        ''')


@pytest.fixture
def dummy_bad_config():
    return yaml.load(
        '''
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


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_initialization(mock_communicator, mock_launcher):
    mock_communicator.return_value = MockCommunicator(
        discrete_action=True, visual_inputs=1)
    tc = TrainerController(' ', ' ', 1, None, True, True, False, 1,
                           1, 1, 1, '', "tests/test_mlagents.trainers.py", False)
    assert (tc.env.brain_names[0] == 'RealFakeBrain')


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_load_config(mock_communicator, mock_launcher, dummy_config):
    open_name = 'mlagents.trainers.trainer_controller' + '.open'
    with mock.patch('yaml.load') as mock_load:
        with mock.patch(open_name, create=True) as _:
            mock_load.return_value = dummy_config
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=1)
            mock_load.return_value = dummy_config
            tc = TrainerController(' ', ' ', 1, None, True, True, False, 1,
                                   1, 1, 1, '', '', False)
            config = tc._load_config()
            assert (len(config) == 1)
            assert (config['default']['trainer'] == "ppo")


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_initialize_trainers(mock_communicator, mock_launcher, dummy_config,
                             dummy_offline_bc_config, dummy_online_bc_config, dummy_bad_config):
    open_name = 'mlagents.trainers.trainer_controller' + '.open'
    with mock.patch('yaml.load') as mock_load:
        with mock.patch(open_name, create=True) as _:
            mock_communicator.return_value = MockCommunicator(
                discrete_action=True, visual_inputs=1)
            tc = TrainerController(' ', ' ', 1, None, True, False, False, 1, 1,
                                   1, 1, '', "tests/test_mlagents.trainers.py",
                                   False)

            # Test for PPO trainer
            mock_load.return_value = dummy_config
            config = tc._load_config()
            tf.reset_default_graph()
            tc._initialize_trainers(config)
            assert (len(tc.trainers) == 1)
            assert (isinstance(tc.trainers['RealFakeBrain'], PPOTrainer))

            # Test for Online Behavior Cloning Trainer
            mock_load.return_value = dummy_online_bc_config
            config = tc._load_config()
            tf.reset_default_graph()
            tc._initialize_trainers(config)
            assert (isinstance(tc.trainers['RealFakeBrain'], OnlineBCTrainer))

            # Test for proper exception when trainer name is incorrect
            mock_load.return_value = dummy_bad_config
            config = tc._load_config()
            tf.reset_default_graph()
            with pytest.raises(UnityEnvironmentException):
                tc._initialize_trainers(config)


@mock.patch('mlagents.envs.UnityEnvironment.executable_launcher')
@mock.patch('mlagents.envs.UnityEnvironment.get_communicator')
def test_initialize_offline_trainers(mock_communicator, mock_launcher, dummy_config,
                             dummy_offline_bc_config, dummy_online_bc_config, dummy_bad_config):
    open_name = 'mlagents.trainers.trainer_controller' + '.open'
    with mock.patch('yaml.load') as mock_load:
        with mock.patch(open_name, create=True) as _:
            mock_communicator.return_value = MockCommunicator(
                discrete_action=False, stack=False, visual_inputs=0,
                brain_name="Ball3DBrain", vec_obs_size=8)
            tc = TrainerController(' ', ' ', 1, None, True, False, False, 1, 1,
                                   1, 1, '', "tests/test_mlagents.trainers.py",
                                   False)

            # Test for Offline Behavior Cloning Trainer
            mock_load.return_value = dummy_offline_bc_config
            config = tc._load_config()
            tf.reset_default_graph()
            tc._initialize_trainers(config)
            assert (isinstance(tc.trainers['Ball3DBrain'], OfflineBCTrainer))

import unittest.mock as mock
import pytest
import mlagents.trainers.tests.mock_brain as mb

import numpy as np
import yaml
import os

from mlagents.trainers.ppo.policy import PPOPolicy


@pytest.fixture
def dummy_config():
    return yaml.safe_load(
        """
        trainer: ppo
        batch_size: 32
        beta: 5.0e-3
        buffer_size: 512
        epsilon: 0.2
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
        pretraining:
          demo_path: ./demos/ExpertPyramid.demo
          strength: 1.0
          steps: 10000000
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


def create_ppo_policy_with_bc_mock(
    mock_env, mock_brain, dummy_config, use_rnn, demo_file
):
    mock_braininfo = mb.create_mock_braininfo(num_agents=12, num_vector_observations=8)
    mb.setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)
    env = mock_env()

    trainer_parameters = dummy_config
    model_path = env.brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["use_recurrent"] = use_rnn
    trainer_parameters["pretraining"]["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/" + demo_file
    )
    policy = PPOPolicy(0, mock_brain, trainer_parameters, False, False)
    return env, policy


# Test default values
@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_defaults(mock_env, dummy_config):
    # See if default values match
    mock_brain = mb.create_mock_3dball_brain()
    env, policy = create_ppo_policy_with_bc_mock(
        mock_env, mock_brain, dummy_config, False, "test.demo"
    )
    assert policy.bc_module.num_epoch == dummy_config["num_epoch"]
    assert policy.bc_module.batch_size == dummy_config["batch_size"]
    env.close()
    # Assign strange values and see if it overrides properly
    dummy_config["pretraining"]["num_epoch"] = 100
    dummy_config["pretraining"]["batch_size"] = 10000
    env, policy = create_ppo_policy_with_bc_mock(
        mock_env, mock_brain, dummy_config, False, "test.demo"
    )
    assert policy.bc_module.num_epoch == 100
    assert policy.bc_module.batch_size == 10000
    env.close()


# Test with continuous control env and vector actions
@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_update(mock_env, dummy_config):
    mock_brain = mb.create_mock_3dball_brain()
    env, policy = create_ppo_policy_with_bc_mock(
        mock_env, mock_brain, dummy_config, False, "test.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


# Test with RNN
@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_rnn_update(mock_env, dummy_config):
    mock_brain = mb.create_mock_3dball_brain()
    env, policy = create_ppo_policy_with_bc_mock(
        mock_env, mock_brain, dummy_config, True, "test.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


# Test with discrete control and visual observations
@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_dc_visual_update(mock_env, dummy_config):
    mock_brain = mb.create_mock_banana_brain()
    env, policy = create_ppo_policy_with_bc_mock(
        mock_env, mock_brain, dummy_config, False, "testdcvis.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


# Test with discrete control, visual observations and RNN
@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_rnn_dc_update(mock_env, dummy_config):
    mock_brain = mb.create_mock_banana_brain()
    env, policy = create_ppo_policy_with_bc_mock(
        mock_env, mock_brain, dummy_config, True, "testdcvis.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


if __name__ == "__main__":
    pytest.main()

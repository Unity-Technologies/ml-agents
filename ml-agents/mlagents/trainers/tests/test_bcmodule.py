from unittest import mock
import pytest
import mlagents.trainers.tests.mock_brain as mb

import numpy as np
import yaml
import os

from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.sac.policy import SACPolicy


def ppo_dummy_config():
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
        behavioral_cloning:
          demo_path: ./demos/ExpertPyramid.demo
          strength: 1.0
          steps: 10000000
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


def sac_dummy_config():
    return yaml.safe_load(
        """
        trainer: sac
        batch_size: 128
        buffer_size: 50000
        buffer_init_steps: 0
        hidden_units: 128
        init_entcoef: 1.0
        learning_rate: 3.0e-4
        max_steps: 5.0e4
        memory_size: 256
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 1000
        tau: 0.005
        use_recurrent: false
        vis_encode_type: simple
        behavioral_cloning:
            demo_path: ./demos/ExpertPyramid.demo
            strength: 1.0
            steps: 10000000
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99
        """
    )


def create_policy_with_bc_mock(
    mock_env, mock_brain, trainer_config, use_rnn, demo_file
):
    mock_braininfo = mb.create_mock_braininfo(num_agents=12, num_vector_observations=8)
    mb.setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)
    env = mock_env()

    model_path = env.external_brain_names[0]
    trainer_config["model_path"] = model_path
    trainer_config["keep_checkpoints"] = 3
    trainer_config["use_recurrent"] = use_rnn
    trainer_config["behavioral_cloning"]["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/" + demo_file
    )

    policy = (
        PPOPolicy(0, mock_brain, trainer_config, False, False)
        if trainer_config["trainer"] == "ppo"
        else SACPolicy(0, mock_brain, trainer_config, False, False)
    )
    return env, policy


# Test default values
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_bcmodule_defaults(mock_env):
    # See if default values match
    mock_brain = mb.create_mock_3dball_brain()
    trainer_config = ppo_dummy_config()
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, False, "test.demo"
    )
    assert policy.bc_module.num_epoch == 3
    assert policy.bc_module.batch_size == trainer_config["batch_size"]
    env.close()
    # Assign strange values and see if it overrides properly
    trainer_config["behavioral_cloning"]["num_epoch"] = 100
    trainer_config["behavioral_cloning"]["batch_size"] = 10000
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, False, "test.demo"
    )
    assert policy.bc_module.num_epoch == 100
    assert policy.bc_module.batch_size == 10000
    env.close()


# Test with continuous control env and vector actions
@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_bcmodule_update(mock_env, trainer_config):
    mock_brain = mb.create_mock_3dball_brain()
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, False, "test.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


# Test with constant pretraining learning rate
@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_bcmodule_constant_lr_update(mock_env, trainer_config):
    mock_brain = mb.create_mock_3dball_brain()
    trainer_config["behavioral_cloning"]["steps"] = 0
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, False, "test.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    old_learning_rate = policy.bc_module.current_lr

    stats = policy.bc_module.update()
    assert old_learning_rate == policy.bc_module.current_lr


# Test with RNN
@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_bcmodule_rnn_update(mock_env, trainer_config):
    mock_brain = mb.create_mock_3dball_brain()
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, True, "test.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


# Test with discrete control and visual observations
@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_bcmodule_dc_visual_update(mock_env, trainer_config):
    mock_brain = mb.create_mock_banana_brain()
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, False, "testdcvis.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


# Test with discrete control, visual observations and RNN
@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
@mock.patch("mlagents_envs.environment.UnityEnvironment")
def test_bcmodule_rnn_dc_update(mock_env, trainer_config):
    mock_brain = mb.create_mock_banana_brain()
    env, policy = create_policy_with_bc_mock(
        mock_env, mock_brain, trainer_config, True, "testdcvis.demo"
    )
    stats = policy.bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    env.close()


if __name__ == "__main__":
    pytest.main()

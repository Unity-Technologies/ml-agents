import unittest.mock as mock
import pytest
import mlagents.trainers.tests.mock_brain as mb

import numpy as np
import tensorflow as tf
import yaml
import os

from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.ppo.trainer import discount_rewards
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.envs import UnityEnvironment


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
          pretraining_strength: 1.0
          pretraining_steps: 10000000
          curiosity_strength: 0.0
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


def create_ppo_policy_with_bc_mock(mock_env, dummy_config, use_rnn):
    mock_brain = mb.create_mock_brainparams(
        vector_action_space_type="continuous",
        vector_action_space_size=[2],
        vector_observation_space_size=8,
    )
    mock_braininfo = mb.create_mock_braininfo(num_agents=12, num_vector_observations=8)
    mb.setup_mock_unityenvironment(mock_env, mock_brain, mock_braininfo)
    env = mock_env()

    trainer_parameters = dummy_config
    model_path = env.brain_names[0]
    trainer_parameters["model_path"] = model_path
    trainer_parameters["keep_checkpoints"] = 3
    trainer_parameters["use_recurrent"] = use_rnn
    trainer_parameters["pretraining"]["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/test.demo"
    )
    policy = PPOPolicy(0, mock_brain, trainer_parameters, False, False)
    return env, policy


@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_update(mock_env, dummy_config):
    env, policy = create_ppo_policy_with_bc_mock(mock_env, dummy_config, False)
    loss = policy.bc_trainer.update()
    assert isinstance(loss, np.float32)
    env.close()


@mock.patch("mlagents.envs.UnityEnvironment")
def test_bcmodule_rnn_update(mock_env, dummy_config):
    env, policy = create_ppo_policy_with_bc_mock(mock_env, dummy_config, True)
    loss = policy.bc_trainer.update()
    assert isinstance(loss, np.float32)
    env.close()


if __name__ == "__main__":
    pytest.main()

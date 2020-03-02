import pytest
import mlagents.trainers.tests.mock_brain as mb

import numpy as np
import yaml
import os

from mlagents.trainers.policy.nn_policy import NNPolicy
from mlagents.trainers.components.bc.module import BCModule


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
          demo_path: ./Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
          strength: 1.0
          steps: 10000000
        reward_signals:
          extrinsic:
            strength: 1.0
            gamma: 0.99
        """
    )


def create_bc_module(mock_brain, trainer_config, use_rnn, demo_file, tanhresample):
    # model_path = env.external_brain_names[0]
    trainer_config["model_path"] = "testpath"
    trainer_config["keep_checkpoints"] = 3
    trainer_config["use_recurrent"] = use_rnn
    trainer_config["behavioral_cloning"]["demo_path"] = (
        os.path.dirname(os.path.abspath(__file__)) + "/" + demo_file
    )

    policy = NNPolicy(
        0, mock_brain, trainer_config, False, False, tanhresample, tanhresample
    )
    with policy.graph.as_default():
        bc_module = BCModule(
            policy,
            policy_learning_rate=trainer_config["learning_rate"],
            default_batch_size=trainer_config["batch_size"],
            default_num_epoch=3,
            **trainer_config["behavioral_cloning"],
        )
    policy.initialize_or_load()  # Normally the optimizer calls this after the BCModule is created
    return bc_module


# Test default values
def test_bcmodule_defaults():
    # See if default values match
    mock_brain = mb.create_mock_3dball_brain()
    trainer_config = ppo_dummy_config()
    bc_module = create_bc_module(mock_brain, trainer_config, False, "test.demo", False)
    assert bc_module.num_epoch == 3
    assert bc_module.batch_size == trainer_config["batch_size"]
    # Assign strange values and see if it overrides properly
    trainer_config["behavioral_cloning"]["num_epoch"] = 100
    trainer_config["behavioral_cloning"]["batch_size"] = 10000
    bc_module = create_bc_module(mock_brain, trainer_config, False, "test.demo", False)
    assert bc_module.num_epoch == 100
    assert bc_module.batch_size == 10000


# Test with continuous control env and vector actions
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_update(is_sac):
    mock_brain = mb.create_mock_3dball_brain()
    bc_module = create_bc_module(
        mock_brain, ppo_dummy_config(), False, "test.demo", is_sac
    )
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


# Test with constant pretraining learning rate
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_constant_lr_update(is_sac):
    trainer_config = ppo_dummy_config()
    mock_brain = mb.create_mock_3dball_brain()
    trainer_config["behavioral_cloning"]["steps"] = 0
    bc_module = create_bc_module(mock_brain, trainer_config, False, "test.demo", is_sac)
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    old_learning_rate = bc_module.current_lr

    stats = bc_module.update()
    assert old_learning_rate == bc_module.current_lr


# Test with RNN
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_rnn_update(is_sac):
    mock_brain = mb.create_mock_3dball_brain()
    bc_module = create_bc_module(
        mock_brain, ppo_dummy_config(), True, "test.demo", is_sac
    )
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


# Test with discrete control and visual observations
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_dc_visual_update(is_sac):
    mock_brain = mb.create_mock_banana_brain()
    bc_module = create_bc_module(
        mock_brain, ppo_dummy_config(), False, "testdcvis.demo", is_sac
    )
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


# Test with discrete control, visual observations and RNN
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_rnn_dc_update(is_sac):
    mock_brain = mb.create_mock_banana_brain()
    bc_module = create_bc_module(
        mock_brain, ppo_dummy_config(), True, "testdcvis.demo", is_sac
    )
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


if __name__ == "__main__":
    pytest.main()

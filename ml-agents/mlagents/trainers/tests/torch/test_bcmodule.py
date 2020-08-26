from unittest.mock import MagicMock
import pytest
import mlagents.trainers.tests.mock_brain as mb

import os

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch.components.bc.module import BCModule
from mlagents.trainers.settings import (
    TrainerSettings,
    BehavioralCloningSettings,
    NetworkSettings,
)


def create_bc_module(mock_behavior_specs, bc_settings, use_rnn, tanhresample):
    # model_path = env.external_brain_names[0]
    trainer_config = TrainerSettings()
    trainer_config.network_settings.memory = (
        NetworkSettings.MemorySettings() if use_rnn else None
    )
    policy = TorchPolicy(
        0, mock_behavior_specs, trainer_config, tanhresample, tanhresample
    )
    bc_module = BCModule(
        policy,
        settings=bc_settings,
        policy_learning_rate=trainer_config.hyperparameters.learning_rate,
        default_batch_size=trainer_config.hyperparameters.batch_size,
        default_num_epoch=3,
    )
    return bc_module


def assert_stats_are_float(stats):
    for _, item in stats.items():
        assert isinstance(item, float)


# Test default values
def test_bcmodule_defaults():
    # See if default values match
    mock_specs = mb.create_mock_3dball_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, False)
    assert bc_module.num_epoch == 3
    assert bc_module.batch_size == TrainerSettings().hyperparameters.batch_size
    # Assign strange values and see if it overrides properly
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo",
        num_epoch=100,
        batch_size=10000,
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, False)
    assert bc_module.num_epoch == 100
    assert bc_module.batch_size == 10000


# Test with continuous control env and vector actions
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_update(is_sac):
    mock_specs = mb.create_mock_3dball_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, is_sac)
    stats = bc_module.update()
    assert_stats_are_float(stats)


# Test with constant pretraining learning rate
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_constant_lr_update(is_sac):
    mock_specs = mb.create_mock_3dball_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo",
        steps=0,
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, is_sac)
    stats = bc_module.update()
    assert_stats_are_float(stats)
    old_learning_rate = bc_module.current_lr

    _ = bc_module.update()
    assert old_learning_rate == bc_module.current_lr


# Test with constant pretraining learning rate
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_linear_lr_update(is_sac):
    mock_specs = mb.create_mock_3dball_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo",
        steps=100,
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, is_sac)
    # Should decay by 10/100 * 0.0003 = 0.00003
    bc_module.policy.get_current_step = MagicMock(return_value=10)
    old_learning_rate = bc_module.current_lr
    _ = bc_module.update()
    assert old_learning_rate - 0.00003 == pytest.approx(bc_module.current_lr, abs=0.01)


# Test with RNN
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_rnn_update(is_sac):
    mock_specs = mb.create_mock_3dball_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, True, is_sac)
    stats = bc_module.update()
    assert_stats_are_float(stats)


# Test with discrete control and visual observations
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_dc_visual_update(is_sac):
    mock_specs = mb.create_mock_banana_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "testdcvis.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, is_sac)
    stats = bc_module.update()
    assert_stats_are_float(stats)


# Test with discrete control, visual observations and RNN
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_rnn_dc_update(is_sac):
    mock_specs = mb.create_mock_banana_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "testdcvis.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, True, is_sac)
    stats = bc_module.update()
    assert_stats_are_float(stats)


if __name__ == "__main__":
    pytest.main()

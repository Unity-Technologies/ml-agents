import pytest
import mlagents.trainers.tests.mock_brain as mb

import numpy as np
import os

from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.components.bc.module import BCModule
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
    policy = TFPolicy(
        0, mock_behavior_specs, trainer_config, tanhresample, tanhresample
    )
    with policy.graph.as_default():
        bc_module = BCModule(
            policy,
            policy_learning_rate=trainer_config.hyperparameters.learning_rate,
            default_batch_size=trainer_config.hyperparameters.batch_size,
            default_num_epoch=3,
            settings=bc_settings,
        )
    policy.initialize()  # Normally the optimizer calls this after the BCModule is created
    return bc_module


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
    for _, item in stats.items():
        assert isinstance(item, np.float32)


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
    for _, item in stats.items():
        assert isinstance(item, np.float32)
    old_learning_rate = bc_module.current_lr

    _ = bc_module.update()
    assert old_learning_rate == bc_module.current_lr


# Test with RNN
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_rnn_update(is_sac):
    mock_specs = mb.create_mock_3dball_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "test.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, True, is_sac)
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


# Test with discrete control and visual observations
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_dc_visual_update(is_sac):
    mock_specs = mb.create_mock_banana_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "testdcvis.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, False, is_sac)
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


# Test with discrete control, visual observations and RNN
@pytest.mark.parametrize("is_sac", [True, False], ids=["sac", "ppo"])
def test_bcmodule_rnn_dc_update(is_sac):
    mock_specs = mb.create_mock_banana_behavior_specs()
    bc_settings = BehavioralCloningSettings(
        demo_path=os.path.dirname(os.path.abspath(__file__)) + "/" + "testdcvis.demo"
    )
    bc_module = create_bc_module(mock_specs, bc_settings, True, is_sac)
    stats = bc_module.update()
    for _, item in stats.items():
        assert isinstance(item, np.float32)


if __name__ == "__main__":
    pytest.main()

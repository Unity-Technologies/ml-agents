import yaml
import pytest
from unittest import mock
from argparse import Namespace

from mlagents.trainers.upgrade_config import convert_behaviors, main, remove_nones
from mlagents.trainers.settings import (
    TrainerType,
    PPOSettings,
    SACSettings,
    RewardSignalType,
)

BRAIN_NAME = "testbehavior"

# Check one per category
BATCH_SIZE = 256
HIDDEN_UNITS = 32
SUMMARY_FREQ = 500

PPO_CONFIG = f"""
    default:
        trainer: ppo
        batch_size: 1024
        beta: 5.0e-3
        buffer_size: 10240
        epsilon: 0.2
        hidden_units: 128
        lambd: 0.95
        learning_rate: 3.0e-4
        learning_rate_schedule: linear
        max_steps: 5.0e5
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 10000
        use_recurrent: false
        vis_encode_type: simple
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99

    {BRAIN_NAME}:
        trainer: ppo
        batch_size: {BATCH_SIZE}
        beta: 5.0e-3
        buffer_size: 64
        epsilon: 0.2
        hidden_units: {HIDDEN_UNITS}
        lambd: 0.95
        learning_rate: 5.0e-3
        max_steps: 2500
        memory_size: 256
        normalize: false
        num_epoch: 3
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: {SUMMARY_FREQ}
        use_recurrent: false
        reward_signals:
            curiosity:
                strength: 1.0
                gamma: 0.99
                encoding_size: 128
    """

SAC_CONFIG = f"""
    default:
        trainer: sac
        batch_size: 128
        buffer_size: 50000
        buffer_init_steps: 0
        hidden_units: 128
        init_entcoef: 1.0
        learning_rate: 3.0e-4
        learning_rate_schedule: constant
        max_steps: 5.0e5
        memory_size: 256
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 2
        time_horizon: 64
        sequence_length: 64
        summary_freq: 10000
        tau: 0.005
        use_recurrent: false
        vis_encode_type: simple
        reward_signals:
            extrinsic:
                strength: 1.0
                gamma: 0.99

    {BRAIN_NAME}:
        trainer: sac
        batch_size: {BATCH_SIZE}
        buffer_size: 64
        buffer_init_steps: 100
        hidden_units: {HIDDEN_UNITS}
        init_entcoef: 0.01
        learning_rate: 3.0e-4
        max_steps: 1000
        memory_size: 256
        normalize: false
        num_update: 1
        train_interval: 1
        num_layers: 1
        time_horizon: 64
        sequence_length: 64
        summary_freq: {SUMMARY_FREQ}
        tau: 0.005
        use_recurrent: false
        curiosity_enc_size: 128
        demo_path: None
        vis_encode_type: simple
        reward_signals:
            curiosity:
                strength: 1.0
                gamma: 0.99
                encoding_size: 128
    """


@pytest.mark.parametrize("use_recurrent", [True, False])
@pytest.mark.parametrize("trainer_type", [TrainerType.PPO, TrainerType.SAC])
def test_convert_behaviors(trainer_type, use_recurrent):
    if trainer_type == TrainerType.PPO:
        trainer_config = PPO_CONFIG
        trainer_settings_type = PPOSettings
    elif trainer_type == TrainerType.SAC:
        trainer_config = SAC_CONFIG
        trainer_settings_type = SACSettings

    old_config = yaml.load(trainer_config)
    old_config[BRAIN_NAME]["use_recurrent"] = use_recurrent
    new_config = convert_behaviors(old_config)

    # Test that the new config can be converted to TrainerSettings w/o exceptions
    trainer_settings = new_config[BRAIN_NAME]

    # Test that the trainer_settings contains the settings for BRAIN_NAME and
    # the defaults where specified
    assert trainer_settings.trainer_type == trainer_type
    assert isinstance(trainer_settings.hyperparameters, trainer_settings_type)
    assert trainer_settings.hyperparameters.batch_size == BATCH_SIZE
    assert trainer_settings.network_settings.hidden_units == HIDDEN_UNITS
    assert RewardSignalType.CURIOSITY in trainer_settings.reward_signals


@mock.patch("mlagents.trainers.upgrade_config.convert_behaviors")
@mock.patch("mlagents.trainers.upgrade_config.remove_nones")
@mock.patch("mlagents.trainers.upgrade_config.write_to_yaml_file")
@mock.patch("mlagents.trainers.upgrade_config.parse_args")
@mock.patch("mlagents.trainers.upgrade_config.load_config")
def test_main(mock_load, mock_parse, yaml_write_mock, remove_none_mock, mock_convert):
    test_output_file = "test.yaml"
    mock_load.side_effect = [
        yaml.safe_load(PPO_CONFIG),
        "test_curriculum_config",
        "test_sampler_config",
    ]
    mock_args = Namespace(
        trainer_config_path="mock",
        output_config_path=test_output_file,
        curriculum="test",
        sampler="test",
    )
    mock_parse.return_value = mock_args
    mock_convert.return_value = "test_converted_config"
    dict_without_nones = mock.Mock(name="nonones")
    remove_none_mock.return_value = dict_without_nones

    main()
    saved_dict = remove_none_mock.call_args[0][0]
    # Check that the output of the remove_none call is here
    yaml_write_mock.assert_called_with(dict_without_nones, test_output_file)
    assert saved_dict["behaviors"] == "test_converted_config"
    assert saved_dict["curriculum"] == "test_curriculum_config"
    assert saved_dict["parameter_randomization"] == "test_sampler_config"


def test_remove_nones():
    dict_with_nones = {"hello": {"hello2": 2, "hello3": None}, "hello4": None}
    dict_without_nones = {"hello": {"hello2": 2}}
    output = remove_nones(dict_with_nones)
    assert output == dict_without_nones

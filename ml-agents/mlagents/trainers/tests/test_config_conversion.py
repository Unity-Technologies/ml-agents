import yaml
import pytest

from mlagents.trainers.upgrade_config import convert_behaviors, remove_nones, convert
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

CURRICULUM = """

  BigWallJump:
    measure: progress
    thresholds: [0.1, 0.3, 0.5]
    min_lesson_length: 200
    signal_smoothing: true
    parameters:
      big_wall_min_height: [0.0, 4.0, 6.0, 8.0]
      big_wall_max_height: [4.0, 7.0, 8.0, 8.0]
  SmallWallJump:
    measure: progress
    thresholds: [0.1, 0.3, 0.5]
    min_lesson_length: 100
    signal_smoothing: true
    parameters:
      small_wall_height: [1.5, 2.0, 2.5, 4.0]
      """

RANDOMIZATION = """
  resampling-interval: 5000
  mass:
    sampler-type: uniform
    min_value: 0.5
    max_value: 10
  gravity:
    sampler-type: uniform
    min_value: 7
    max_value: 12
  scale:
    sampler-type: uniform
    min_value: 0.75
    max_value: 3
    """


@pytest.mark.parametrize("use_recurrent", [True, False])
@pytest.mark.parametrize("trainer_type", [TrainerType.PPO, TrainerType.SAC])
def test_convert_behaviors(trainer_type, use_recurrent):
    if trainer_type == TrainerType.PPO:
        trainer_config = PPO_CONFIG
        trainer_settings_type = PPOSettings
    else:
        trainer_config = SAC_CONFIG
        trainer_settings_type = SACSettings

    old_config = yaml.safe_load(trainer_config)
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


def test_convert():
    old_behaviors = yaml.safe_load(PPO_CONFIG)
    old_curriculum = yaml.safe_load(CURRICULUM)
    old_sampler = yaml.safe_load(RANDOMIZATION)
    config = convert(old_behaviors, old_curriculum, old_sampler)
    assert BRAIN_NAME in config["behaviors"]
    assert "big_wall_min_height" in config["environment_parameters"]

    curriculum = config["environment_parameters"]["big_wall_min_height"]["curriculum"]
    assert len(curriculum) == 4
    for i, expected_value in enumerate([0.0, 4.0, 6.0, 8.0]):
        assert curriculum[i][f"Lesson{i}"]["value"] == expected_value
    for i, threshold in enumerate([0.1, 0.3, 0.5]):
        criteria = curriculum[i][f"Lesson{i}"]["completion_criteria"]
        assert criteria["threshold"] == threshold
        assert criteria["behavior"] == "BigWallJump"
        assert criteria["signal_smoothing"]
        assert criteria["min_lesson_length"] == 200
        assert criteria["measure"] == "progress"

    assert "gravity" in config["environment_parameters"]
    gravity = config["environment_parameters"]["gravity"]
    assert gravity["sampler_type"] == "uniform"
    assert gravity["sampler_parameters"]["min_value"] == 7
    assert gravity["sampler_parameters"]["max_value"] == 12


def test_remove_nones():
    dict_with_nones = {"hello": {"hello2": 2, "hello3": None}, "hello4": None}
    dict_without_nones = {"hello": {"hello2": 2}}
    output = remove_nones(dict_with_nones)
    assert output == dict_without_nones

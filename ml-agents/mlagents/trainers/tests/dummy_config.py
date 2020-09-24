import pytest
import copy
import os
from mlagents.trainers.settings import (
    TrainerSettings,
    PPOSettings,
    SACSettings,
    GAILSettings,
    CuriositySettings,
    RewardSignalSettings,
    NetworkSettings,
    TrainerType,
    RewardSignalType,
    ScheduleType,
)

CONTINUOUS_DEMO_PATH = os.path.dirname(os.path.abspath(__file__)) + "/test.demo"
DISCRETE_DEMO_PATH = os.path.dirname(os.path.abspath(__file__)) + "/testdcvis.demo"

_PPO_CONFIG = TrainerSettings(
    trainer_type=TrainerType.PPO,
    hyperparameters=PPOSettings(
        learning_rate=5.0e-3,
        learning_rate_schedule=ScheduleType.CONSTANT,
        batch_size=16,
        buffer_size=64,
    ),
    network_settings=NetworkSettings(num_layers=1, hidden_units=32),
    summary_freq=500,
    max_steps=3000,
    threaded=False,
)

_SAC_CONFIG = TrainerSettings(
    trainer_type=TrainerType.SAC,
    hyperparameters=SACSettings(
        learning_rate=5.0e-3,
        learning_rate_schedule=ScheduleType.CONSTANT,
        batch_size=8,
        buffer_init_steps=100,
        buffer_size=5000,
        tau=0.01,
        init_entcoef=0.01,
    ),
    network_settings=NetworkSettings(num_layers=1, hidden_units=16),
    summary_freq=100,
    max_steps=1000,
    threaded=False,
)


def ppo_dummy_config():
    return copy.deepcopy(_PPO_CONFIG)


def sac_dummy_config():
    return copy.deepcopy(_SAC_CONFIG)


@pytest.fixture
def gail_dummy_config():
    return {RewardSignalType.GAIL: GAILSettings(demo_path=CONTINUOUS_DEMO_PATH)}


@pytest.fixture
def curiosity_dummy_config():
    return {RewardSignalType.CURIOSITY: CuriositySettings()}


@pytest.fixture
def extrinsic_dummy_config():
    return {RewardSignalType.EXTRINSIC: RewardSignalSettings()}

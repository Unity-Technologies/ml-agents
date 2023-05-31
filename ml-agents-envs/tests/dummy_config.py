from typing import List, Tuple
from mlagents_envs.base_env import ObservationSpec, DimensionProperty, ObservationType
import pytest
import copy
import os
from mlagents.trainers.settings import (
    TrainerSettings,
    GAILSettings,
    CuriositySettings,
    RewardSignalSettings,
    NetworkSettings,
    RewardSignalType,
    ScheduleType,
)
from mlagents.trainers.ppo.trainer import PPOSettings, TRAINER_NAME as PPO_TRAINER_NAME
from mlagents.trainers.sac.trainer import SACSettings, TRAINER_NAME as SAC_TRAINER_NAME
from mlagents.trainers.poca.trainer import (
    POCASettings,
    TRAINER_NAME as POCA_TRAINER_NAME,
)

CONTINUOUS_DEMO_PATH = os.path.dirname(os.path.abspath(__file__)) + "/test.demo"
DISCRETE_DEMO_PATH = os.path.dirname(os.path.abspath(__file__)) + "/testdcvis.demo"

_PPO_CONFIG = TrainerSettings(
    trainer_type=PPO_TRAINER_NAME,
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
    trainer_type=SAC_TRAINER_NAME,
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

_POCA_CONFIG = TrainerSettings(
    trainer_type=POCA_TRAINER_NAME,
    hyperparameters=POCASettings(
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


def ppo_dummy_config():
    return copy.deepcopy(_PPO_CONFIG)


def sac_dummy_config():
    return copy.deepcopy(_SAC_CONFIG)


def poca_dummy_config():
    return copy.deepcopy(_POCA_CONFIG)


@pytest.fixture
def gail_dummy_config():
    return {RewardSignalType.GAIL: GAILSettings(demo_path=CONTINUOUS_DEMO_PATH)}


@pytest.fixture
def curiosity_dummy_config():
    return {RewardSignalType.CURIOSITY: CuriositySettings()}


@pytest.fixture
def extrinsic_dummy_config():
    return {RewardSignalType.EXTRINSIC: RewardSignalSettings()}


def create_observation_specs_with_shapes(
    shapes: List[Tuple[int, ...]]
) -> List[ObservationSpec]:
    obs_specs: List[ObservationSpec] = []
    for i, shape in enumerate(shapes):
        dim_prop = (DimensionProperty.UNSPECIFIED,) * len(shape)
        if len(shape) == 2:
            dim_prop = (DimensionProperty.VARIABLE_SIZE, DimensionProperty.NONE)
        spec = ObservationSpec(
            name=f"observation {i} with shape {shape}",
            shape=shape,
            dimension_property=dim_prop,
            observation_type=ObservationType.DEFAULT,
        )
        obs_specs.append(spec)
    return obs_specs

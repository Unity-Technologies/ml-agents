import attr
import pytest


from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
    MemoryEnvironment,
)

from mlagents.trainers.settings import NetworkSettings, FrameworkType

from mlagents.trainers.tests.dummy_config import ppo_dummy_config, sac_dummy_config
from mlagents.trainers.tests.check_env_trains import check_environment_trains

BRAIN_NAME = "1D"

PPO_TORCH_CONFIG = attr.evolve(ppo_dummy_config(), framework=FrameworkType.PYTORCH)
SAC_TORCH_CONFIG = attr.evolve(sac_dummy_config(), framework=FrameworkType.PYTORCH)


def test_hybrid_ppo():
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=(1, 1))
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_ppo(num_visual):
    env = SimpleEnvironment(
        [BRAIN_NAME], num_visual=num_visual, num_vector=0, action_sizes=(1, 1)
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


def test_recurrent_ppo():
    env = MemoryEnvironment([BRAIN_NAME], action_sizes=(1, 1))
    new_network_settings = attr.evolve(
        PPO_TORCH_CONFIG.network_settings,
        memory=NetworkSettings.MemorySettings(memory_size=16),
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters,
        learning_rate=1.0e-3,
        batch_size=64,
        buffer_size=128,
    )
    config = attr.evolve(
        PPO_TORCH_CONFIG,
        hyperparameters=new_hyperparams,
        network_settings=new_network_settings,
        max_steps=10000,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)


@pytest.mark.parametrize("action_size", [(1, 1), (2, 2)])
def test_hybrid_sac(action_size):
    env = SimpleEnvironment([BRAIN_NAME], action_sizes=action_size)

    new_hyperparams = attr.evolve(
        SAC_TORCH_CONFIG.hyperparameters, buffer_size=50000, batch_size=128
    )
    config = attr.evolve(
        SAC_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=3000
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)

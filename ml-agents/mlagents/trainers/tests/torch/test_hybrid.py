import pytest
import attr


from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
    MemoryEnvironment,
    RecordEnvironment,
)

from mlagents.trainers.demo_loader import write_demo

from mlagents.trainers.settings import (
    NetworkSettings,
    SelfPlaySettings,
    BehavioralCloningSettings,
    GAILSettings,
    RewardSignalType,
    EncoderType,
    FrameworkType,
)

from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.space_type_pb2 import discrete, continuous

from mlagents.trainers.tests.dummy_config import ppo_dummy_config, sac_dummy_config
from mlagents.trainers.tests.check_env_trains import (
    check_environment_trains,
    default_reward_processor,
)

BRAIN_NAME = "1D"

PPO_TORCH_CONFIG = attr.evolve(ppo_dummy_config(), framework=FrameworkType.PYTORCH)
SAC_TORCH_CONFIG = attr.evolve(sac_dummy_config(), framework=FrameworkType.PYTORCH)


def test_hybrid_ppo():
    env = SimpleEnvironment(
        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=1
    )
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


def test_conthybrid_ppo():
    env = SimpleEnvironment(
        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=0
    )
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


def test_dischybrid_ppo():
    env = SimpleEnvironment(
        [BRAIN_NAME], continuous_action_size=0, discrete_action_size=1
    )
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


@pytest.mark.parametrize("num_visual", [1, 2])
def test_visual_ppo(num_visual):
    env = SimpleEnvironment(
        [BRAIN_NAME],
        num_visual=num_visual,
        num_vector=0,
        continuous_action_size=1,
        discrete_action_size=1,
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, learning_rate=3.0e-4
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams)
    check_environment_trains(env, {BRAIN_NAME: config})


def test_recurrent_ppo():
    env = MemoryEnvironment(
        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=1
    )
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
        max_steps=100000,
    )
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=0.9)


# def test_3cdhybrid_ppo():
#    env = SimpleEnvironment(
#        [BRAIN_NAME], continuous_action_size=2, discrete_action_size=1, step_size=0.8
#    )
#    new_hyperparams = attr.evolve(
#        PPO_TORCH_CONFIG.hyperparameters, batch_size=128, buffer_size=1280, beta=0.01
#    )
#    config = attr.evolve(
#        PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000
#    )
#    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)
#
#
# def test_3ddhybrid_ppo():
#    env = SimpleEnvironment(
#        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=2, step_size=0.8
#    )
#    new_hyperparams = attr.evolve(
#        PPO_TORCH_CONFIG.hyperparameters, batch_size=128, buffer_size=1280, beta=0.01
#    )
#    config = attr.evolve(
#        PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000
#    )
#    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)

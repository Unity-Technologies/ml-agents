import pytest
import attr


from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
    HybridEnvironment,
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

# @pytest.mark.parametrize("use_discrete", [True, False])
# def test_simple_ppo(use_discrete):
#    env = SimpleEnvironment([BRAIN_NAME], use_discrete=use_discrete)
#    config = attr.evolve(PPO_TORCH_CONFIG)
#    _check_environment_trains(env, {BRAIN_NAME: config})


def test_hybrid_ppo():
    env = HybridEnvironment(
        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=1, step_size=0.8
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, batch_size=32, buffer_size=1280
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


def test_conthybrid_ppo():
    env = HybridEnvironment(
        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=0, step_size=0.8
    )
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


def test_dischybrid_ppo():
    env = HybridEnvironment(
        [BRAIN_NAME], continuous_action_size=0, discrete_action_size=1, step_size=0.8
    )
    config = attr.evolve(PPO_TORCH_CONFIG)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


def test_3cdhybrid_ppo():
    env = HybridEnvironment([BRAIN_NAME], continuous_action_size=2, discrete_action_size=1, step_size=0.8)
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, batch_size=128, buffer_size=1280, beta=0.01
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


def test_3ddhybrid_ppo():
    env = HybridEnvironment(
        [BRAIN_NAME], continuous_action_size=1, discrete_action_size=2, step_size=0.8
    )
    new_hyperparams = attr.evolve(
        PPO_TORCH_CONFIG.hyperparameters, batch_size=128, buffer_size=1280, beta=0.01
    )
    config = attr.evolve(PPO_TORCH_CONFIG, hyperparameters=new_hyperparams, max_steps=10000)
    check_environment_trains(env, {BRAIN_NAME: config}, success_threshold=1.0)


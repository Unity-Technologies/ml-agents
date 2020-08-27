import pytest
import copy
import os
import mlagents.trainers.tests.mock_brain as mb
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.sac.optimizer_tf import SACOptimizer
from mlagents.trainers.ppo.optimizer_tf import PPOOptimizer
from mlagents.trainers.tests.test_simple_rl import PPO_CONFIG, SAC_CONFIG
from mlagents.trainers.settings import (
    GAILSettings,
    CuriositySettings,
    RewardSignalSettings,
    BehavioralCloningSettings,
    NetworkSettings,
    TrainerType,
    RewardSignalType,
)

CONTINUOUS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/test.demo"
DISCRETE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/testdcvis.demo"


def ppo_dummy_config():
    return copy.deepcopy(PPO_CONFIG)


def sac_dummy_config():
    return copy.deepcopy(SAC_CONFIG)


@pytest.fixture
def gail_dummy_config():
    return {RewardSignalType.GAIL: GAILSettings(demo_path=CONTINUOUS_PATH)}


@pytest.fixture
def curiosity_dummy_config():
    return {RewardSignalType.CURIOSITY: CuriositySettings()}


@pytest.fixture
def extrinsic_dummy_config():
    return {RewardSignalType.EXTRINSIC: RewardSignalSettings()}


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 20
BATCH_SIZE = 12
NUM_AGENTS = 12


def create_optimizer_mock(
    trainer_config, reward_signal_config, use_rnn, use_discrete, use_visual
):
    mock_specs = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE if not use_visual else 0,
    )
    trainer_settings = trainer_config
    trainer_settings.reward_signals = reward_signal_config
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=10)
        if use_rnn
        else None
    )
    policy = TFPolicy(
        0, mock_specs, trainer_settings, "test", False, create_tf_graph=False
    )
    if trainer_settings.trainer_type == TrainerType.SAC:
        optimizer = SACOptimizer(policy, trainer_settings)
    else:
        optimizer = PPOOptimizer(policy, trainer_settings)
    optimizer.policy.initialize()
    return optimizer


def reward_signal_eval(optimizer, reward_signal_name):
    buffer = mb.simulate_rollout(BATCH_SIZE, optimizer.policy.behavior_spec)
    # Test evaluate
    rsig_result = optimizer.reward_signals[reward_signal_name].evaluate_batch(buffer)
    assert rsig_result.scaled_reward.shape == (BATCH_SIZE,)
    assert rsig_result.unscaled_reward.shape == (BATCH_SIZE,)


def reward_signal_update(optimizer, reward_signal_name):
    buffer = mb.simulate_rollout(BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec)
    feed_dict = optimizer.reward_signals[reward_signal_name].prepare_update(
        optimizer.policy, buffer.make_mini_batch(0, 10), 2
    )
    out = optimizer.policy._execute_model(
        feed_dict, optimizer.reward_signals[reward_signal_name].update_dict
    )
    assert type(out) is dict


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_gail_cc(trainer_config, gail_dummy_config):
    trainer_config.behavioral_cloning = BehavioralCloningSettings(
        demo_path=CONTINUOUS_PATH
    )
    optimizer = create_optimizer_mock(
        trainer_config, gail_dummy_config, False, False, False
    )
    reward_signal_eval(optimizer, "gail")
    reward_signal_update(optimizer, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_gail_dc_visual(trainer_config, gail_dummy_config):
    gail_dummy_config_discrete = {
        RewardSignalType.GAIL: GAILSettings(demo_path=DISCRETE_PATH)
    }
    optimizer = create_optimizer_mock(
        trainer_config, gail_dummy_config_discrete, False, True, True
    )
    reward_signal_eval(optimizer, "gail")
    reward_signal_update(optimizer, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_gail_rnn(trainer_config, gail_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, gail_dummy_config, True, False, False
    )
    reward_signal_eval(policy, "gail")
    reward_signal_update(policy, "gail")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_cc(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, False, False
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_dc(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, True, False
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_visual(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, False, False, True
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_curiosity_rnn(trainer_config, curiosity_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, curiosity_dummy_config, True, False, False
    )
    reward_signal_eval(policy, "curiosity")
    reward_signal_update(policy, "curiosity")


@pytest.mark.parametrize(
    "trainer_config", [ppo_dummy_config(), sac_dummy_config()], ids=["ppo", "sac"]
)
def test_extrinsic(trainer_config, extrinsic_dummy_config):
    policy = create_optimizer_mock(
        trainer_config, extrinsic_dummy_config, False, False, False
    )
    reward_signal_eval(policy, "extrinsic")
    reward_signal_update(policy, "extrinsic")


if __name__ == "__main__":
    pytest.main()

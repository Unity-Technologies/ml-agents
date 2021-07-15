from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
import pytest
from unittest import mock
from typing import cast
from mlagents.torch_utils import torch

from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.settings import NetworkSettings, TrainerSettings
from mlagents.trainers.tests.dummy_config import (  # noqa: F401
    sac_dummy_config,
    curiosity_dummy_config,
)


@pytest.fixture
def dummy_config() -> TrainerSettings:
    return sac_dummy_config()


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12


def create_sac_optimizer_mock(dummy_config, use_rnn, use_discrete, use_visual):
    mock_brain = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE if not use_visual else 0,
    )
    trainer_settings = dummy_config
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=12)
        if use_rnn
        else None
    )
    policy = TorchPolicy(0, mock_brain, trainer_settings)
    optimizer = TorchSACOptimizer(policy, trainer_settings)
    return optimizer


def create_sac_trainer(
    dummy_config: TrainerSettings, use_rnn: bool = False
) -> SACTrainer:
    mock_brain = mb.setup_test_behavior_specs(
        True,
        False,
        vector_action_space=DISCRETE_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )
    trainer_settings = dummy_config
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=12)
        if use_rnn
        else None
    )
    tset = {"test": trainer_settings}
    mock_param_manager = mock.Mock()
    mock_param_manager.get_minimum_reward_buffer_size.return_value = 1
    trainer_factory = TrainerFactory(
        trainer_config=tset,
        output_path="./",
        train_model=True,
        load_model=False,
        seed=0,
        param_manager=mock_param_manager,
    )
    trainer = trainer_factory.generate("test")
    bid = BehaviorIdentifiers("test", "test", 0)
    policy = trainer.create_policy(bid, mock_brain)
    trainer = cast(SACTrainer, trainer.add_policy(bid, policy))
    return trainer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_sac_optimizer_update(dummy_config, rnn, visual, discrete):
    torch.manual_seed(0)
    # Test evaluate
    optimizer = create_sac_optimizer_mock(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec, memory_size=12
    )
    # Mock out reward signal eval
    update_buffer[RewardSignalUtil.rewards_key("extrinsic")] = update_buffer[
        BufferKey.ENVIRONMENT_REWARDS
    ]
    # Mock out value memories
    update_buffer[BufferKey.CRITIC_MEMORY] = update_buffer[BufferKey.MEMORY]
    return_stats = optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )
    # Make sure we have the right stats
    required_stats = [
        "Losses/Policy Loss",
        "Losses/Value Loss",
        "Losses/Q1 Loss",
        "Losses/Q2 Loss",
        "Policy/Continuous Entropy Coeff",
        "Policy/Discrete Entropy Coeff",
        "Policy/Learning Rate",
    ]
    for stat in required_stats:
        assert stat in return_stats.keys()


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
def test_sac_update_reward_signals(
    dummy_config, curiosity_dummy_config, discrete  # noqa: F811
):
    # Add a Curiosity module
    dummy_config.reward_signals = curiosity_dummy_config
    optimizer = create_sac_optimizer_mock(
        dummy_config, use_rnn=False, use_discrete=discrete, use_visual=False
    )

    # Test update, while removing PPO-specific buffer elements.
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )

    # Mock out reward signal eval
    update_buffer[RewardSignalUtil.rewards_key("extrinsic")] = update_buffer[
        BufferKey.ENVIRONMENT_REWARDS
    ]
    update_buffer[RewardSignalUtil.rewards_key("curiosity")] = update_buffer[
        BufferKey.ENVIRONMENT_REWARDS
    ]
    return_stats = optimizer.update_reward_signals(
        {"curiosity": update_buffer}, num_sequences=update_buffer.num_experiences
    )
    required_stats = ["Losses/Curiosity Forward Loss", "Losses/Curiosity Inverse Loss"]
    for stat in required_stats:
        assert stat in return_stats.keys()


def test_sac_buffer_truncate(dummy_config: TrainerSettings) -> None:
    new_config = dummy_config
    # Weird buffer size
    new_config.hyperparameters.buffer_size = 1234
    sac_trainer = create_sac_trainer(dummy_config, use_rnn=True)
    sac_trainer.update_buffer = mb.simulate_rollout(
        2000, sac_trainer.optimizer.policy.behavior_spec
    )
    sac_trainer._update_policy()
    # Check to make sure buffer has been truncated properly
    max_buffer_size = dummy_config.hyperparameters.buffer_size
    assert sac_trainer.update_buffer.num_experiences < max_buffer_size
    # Make sure it is a multiple of sequence length in there
    assert (
        sac_trainer.update_buffer.num_experiences % sac_trainer.policy.sequence_length
        == 0
    )


if __name__ == "__main__":
    pytest.main()

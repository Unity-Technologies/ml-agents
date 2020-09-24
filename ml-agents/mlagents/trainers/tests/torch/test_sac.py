import pytest
from mlagents.torch_utils import torch
import attr

from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.settings import NetworkSettings, FrameworkType
from mlagents.trainers.tests.dummy_config import (  # noqa: F401; pylint: disable=unused-variable
    sac_dummy_config,
    curiosity_dummy_config,
)


@pytest.fixture
def dummy_config():
    return attr.evolve(sac_dummy_config(), framework=FrameworkType.PYTORCH)


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
    policy = TorchPolicy(0, mock_brain, trainer_settings, "test", False)
    optimizer = TorchSACOptimizer(policy, trainer_settings)
    return optimizer


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
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec, memory_size=24
    )
    # Mock out reward signal eval
    update_buffer["extrinsic_rewards"] = update_buffer["environment_rewards"]
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
        "Policy/Entropy Coeff",
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
    update_buffer["extrinsic_rewards"] = update_buffer["environment_rewards"]
    update_buffer["curiosity_rewards"] = update_buffer["environment_rewards"]
    return_stats = optimizer.update_reward_signals(
        {"curiosity": update_buffer}, num_sequences=update_buffer.num_experiences
    )
    required_stats = ["Losses/Curiosity Forward Loss", "Losses/Curiosity Inverse Loss"]
    for stat in required_stats:
        assert stat in return_stats.keys()


if __name__ == "__main__":
    pytest.main()

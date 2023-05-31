import pytest

import numpy as np
import attr

from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import copy_buffer_fields
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.tests.dummy_config import (  # noqa: F401
    ppo_dummy_config,
    curiosity_dummy_config,
    gail_dummy_config,
)

from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil


@pytest.fixture
def dummy_config():
    return ppo_dummy_config()


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 12

CONTINUOUS_ACTION_SPEC = ActionSpec.create_continuous(VECTOR_ACTION_SPACE)
DISCRETE_ACTION_SPEC = ActionSpec.create_discrete(tuple(DISCRETE_ACTION_SPACE))


def create_test_ppo_optimizer(dummy_config, use_rnn, use_discrete, use_visual):
    mock_specs = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )

    trainer_settings = attr.evolve(dummy_config)
    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=16, memory_size=10)
        if use_rnn
        else None
    )
    actor_kwargs = {
        "conditional_sigma": False,
        "tanh_squash": False,
    }
    policy = TorchPolicy(
        0, mock_specs, trainer_settings.network_settings, SimpleActor, actor_kwargs
    )
    optimizer = TorchPPOOptimizer(policy, trainer_settings)
    return optimizer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_ppo_optimizer_update(dummy_config, rnn, visual, discrete):
    # Test evaluate
    optimizer = create_test_ppo_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES,
        optimizer.policy.behavior_spec,
        memory_size=optimizer.policy.m_size,
    )
    # Mock out reward signal eval
    copy_buffer_fields(
        update_buffer,
        BufferKey.ENVIRONMENT_REWARDS,
        [
            BufferKey.ADVANTAGES,
            RewardSignalUtil.returns_key("extrinsic"),
            RewardSignalUtil.value_estimates_key("extrinsic"),
        ],
    )
    # Copy memories to critic memories
    copy_buffer_fields(update_buffer, BufferKey.MEMORY, [BufferKey.CRITIC_MEMORY])

    return_stats = optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )
    # Make sure we have the right stats
    required_stats = [
        "Losses/Policy Loss",
        "Losses/Value Loss",
        "Policy/Learning Rate",
        "Policy/Epsilon",
        "Policy/Beta",
    ]
    for stat in required_stats:
        assert stat in return_stats.keys()


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_ppo_optimizer_update_curiosity(
    dummy_config, curiosity_dummy_config, rnn, visual, discrete  # noqa: F811
):
    # Test evaluate
    dummy_config.reward_signals = curiosity_dummy_config
    optimizer = create_test_ppo_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES,
        optimizer.policy.behavior_spec,
        memory_size=optimizer.policy.m_size,
    )
    # Mock out reward signal eval
    copy_buffer_fields(
        update_buffer,
        src_key=BufferKey.ENVIRONMENT_REWARDS,
        dst_keys=[
            BufferKey.ADVANTAGES,
            RewardSignalUtil.returns_key("extrinsic"),
            RewardSignalUtil.value_estimates_key("extrinsic"),
            RewardSignalUtil.returns_key("curiosity"),
            RewardSignalUtil.value_estimates_key("curiosity"),
        ],
    )
    # Copy memories to critic memories
    copy_buffer_fields(update_buffer, BufferKey.MEMORY, [BufferKey.CRITIC_MEMORY])

    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_ppo_optimizer_update_gail(gail_dummy_config, dummy_config):  # noqa: F811
    # Test evaluate
    dummy_config.reward_signals = gail_dummy_config
    config = ppo_dummy_config()
    optimizer = create_test_ppo_optimizer(
        config, use_rnn=False, use_discrete=False, use_visual=False
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES, optimizer.policy.behavior_spec
    )
    # Mock out reward signal eval
    copy_buffer_fields(
        update_buffer,
        src_key=BufferKey.ENVIRONMENT_REWARDS,
        dst_keys=[
            BufferKey.ADVANTAGES,
            RewardSignalUtil.returns_key("extrinsic"),
            RewardSignalUtil.value_estimates_key("extrinsic"),
            RewardSignalUtil.returns_key("gail"),
            RewardSignalUtil.value_estimates_key("gail"),
        ],
    )

    update_buffer[BufferKey.CONTINUOUS_LOG_PROBS] = np.ones_like(
        update_buffer[BufferKey.CONTINUOUS_ACTION]
    )
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )

    # Check if buffer size is too big
    update_buffer = mb.simulate_rollout(3000, optimizer.policy.behavior_spec)
    # Mock out reward signal eval
    copy_buffer_fields(
        update_buffer,
        src_key=BufferKey.ENVIRONMENT_REWARDS,
        dst_keys=[
            BufferKey.ADVANTAGES,
            RewardSignalUtil.returns_key("extrinsic"),
            RewardSignalUtil.value_estimates_key("extrinsic"),
            RewardSignalUtil.returns_key("gail"),
            RewardSignalUtil.value_estimates_key("gail"),
        ],
    )
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_ppo_get_value_estimates(dummy_config, rnn, visual, discrete):
    optimizer = create_test_ppo_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Time horizon is longer than sequence length, make sure to test
    # process trajectory on multiple sequences in trajectory + some padding
    time_horizon = 30
    trajectory = make_fake_trajectory(
        length=time_horizon,
        observation_specs=optimizer.policy.behavior_spec.observation_specs,
        action_spec=DISCRETE_ACTION_SPEC if discrete else CONTINUOUS_ACTION_SPEC,
        max_step_complete=True,
    )
    run_out, final_value_out, all_memories = optimizer.get_trajectory_value_estimates(
        trajectory.to_agentbuffer(), trajectory.next_obs, done=False
    )
    if rnn:
        # Check that memories don't have a Torch gradient
        for mem in optimizer.critic_memory_dict.values():
            assert not mem.requires_grad

    for key, val in run_out.items():
        assert type(key) is str
        assert len(val) == time_horizon
    if all_memories is not None:
        assert len(all_memories) == time_horizon

    run_out, final_value_out, _ = optimizer.get_trajectory_value_estimates(
        trajectory.to_agentbuffer(), trajectory.next_obs, done=True
    )
    for key, val in final_value_out.items():
        assert type(key) is str
        assert val == 0.0

    # Check if we ignore terminal states properly
    optimizer.reward_signals["extrinsic"].use_terminal_states = False
    run_out, final_value_out, _ = optimizer.get_trajectory_value_estimates(
        trajectory.to_agentbuffer(), trajectory.next_obs, done=False
    )
    for key, val in final_value_out.items():
        assert type(key) is str
        assert val != 0.0


if __name__ == "__main__":
    pytest.main()

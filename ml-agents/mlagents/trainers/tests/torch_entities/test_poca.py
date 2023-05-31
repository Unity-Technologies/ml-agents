from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
import pytest
from typing import Dict, Any
import numpy as np
import attr

# Import to avoid circular import
from mlagents.trainers.trainer.trainer_factory import TrainerFactory  # noqa F401

from mlagents.trainers.poca.optimizer_torch import TorchPOCAOptimizer
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import copy_buffer_fields
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.tests.dummy_config import (  # noqa: F401
    create_observation_specs_with_shapes,
    poca_dummy_config,
    curiosity_dummy_config,
    gail_dummy_config,
)
from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.settings import TrainerSettings

from mlagents_envs.base_env import ActionSpec, BehaviorSpec
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil


@pytest.fixture
def dummy_config():
    return poca_dummy_config()


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 4

CONTINUOUS_ACTION_SPEC = ActionSpec.create_continuous(VECTOR_ACTION_SPACE)
DISCRETE_ACTION_SPEC = ActionSpec.create_discrete(tuple(DISCRETE_ACTION_SPACE))


def create_test_poca_optimizer(dummy_config, use_rnn, use_discrete, use_visual):
    mock_specs = mb.setup_test_behavior_specs(
        use_discrete,
        use_visual,
        vector_action_space=DISCRETE_ACTION_SPACE
        if use_discrete
        else VECTOR_ACTION_SPACE,
        vector_obs_space=VECTOR_OBS_SPACE,
    )

    trainer_settings = attr.evolve(dummy_config)
    trainer_settings.reward_signals = {
        RewardSignalType.EXTRINSIC: RewardSignalSettings(strength=1.0, gamma=0.99)
    }

    trainer_settings.network_settings.memory = (
        NetworkSettings.MemorySettings(sequence_length=8, memory_size=10)
        if use_rnn
        else None
    )
    actor_kwargs: Dict[str, Any] = {
        "conditional_sigma": False,
        "tanh_squash": False,
    }
    policy = TorchPolicy(
        0, mock_specs, trainer_settings.network_settings, SimpleActor, actor_kwargs
    )
    optimizer = TorchPOCAOptimizer(policy, trainer_settings)
    return optimizer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_poca_optimizer_update(dummy_config, rnn, visual, discrete):
    # Test evaluate
    optimizer = create_test_poca_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    # Test update
    update_buffer = mb.simulate_rollout(
        BUFFER_INIT_SAMPLES,
        optimizer.policy.behavior_spec,
        memory_size=optimizer.policy.m_size,
        num_other_agents_in_group=NUM_AGENTS,
    )
    # Mock out reward signal eval
    copy_buffer_fields(
        update_buffer,
        BufferKey.ENVIRONMENT_REWARDS,
        [
            BufferKey.ADVANTAGES,
            RewardSignalUtil.returns_key("extrinsic"),
            RewardSignalUtil.value_estimates_key("extrinsic"),
            RewardSignalUtil.baseline_estimates_key("extrinsic"),
        ],
    )
    # Copy memories to critic memories
    copy_buffer_fields(
        update_buffer,
        BufferKey.MEMORY,
        [BufferKey.CRITIC_MEMORY, BufferKey.BASELINE_MEMORY],
    )

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
def test_poca_get_value_estimates(dummy_config, rnn, visual, discrete):
    optimizer = create_test_poca_optimizer(
        dummy_config, use_rnn=rnn, use_discrete=discrete, use_visual=visual
    )
    time_horizon = 30
    trajectory = make_fake_trajectory(
        length=time_horizon,
        observation_specs=optimizer.policy.behavior_spec.observation_specs,
        action_spec=DISCRETE_ACTION_SPEC if discrete else CONTINUOUS_ACTION_SPEC,
        max_step_complete=True,
        num_other_agents_in_group=NUM_AGENTS,
    )
    (
        value_estimates,
        baseline_estimates,
        value_next,
        value_memories,
        baseline_memories,
    ) = optimizer.get_trajectory_and_baseline_value_estimates(
        trajectory.to_agentbuffer(),
        trajectory.next_obs,
        trajectory.next_group_obs,
        done=False,
    )
    for key, val in value_estimates.items():
        assert type(key) is str
        assert len(val) == time_horizon
    for key, val in baseline_estimates.items():
        assert type(key) is str
        assert len(val) == time_horizon

    if value_memories is not None:
        assert len(value_memories) == time_horizon
        assert len(baseline_memories) == time_horizon

    (
        value_estimates,
        baseline_estimates,
        value_next,
        value_memories,
        baseline_memories,
    ) = optimizer.get_trajectory_and_baseline_value_estimates(
        trajectory.to_agentbuffer(),
        trajectory.next_obs,
        trajectory.next_group_obs,
        done=True,
    )
    for key, val in value_next.items():
        assert type(key) is str
        assert val == 0.0

    # Check if we ignore terminal states properly
    optimizer.reward_signals["extrinsic"].use_terminal_states = False
    (
        value_estimates,
        baseline_estimates,
        value_next,
        value_memories,
        baseline_memories,
    ) = optimizer.get_trajectory_and_baseline_value_estimates(
        trajectory.to_agentbuffer(),
        trajectory.next_obs,
        trajectory.next_group_obs,
        done=False,
    )
    for key, val in value_next.items():
        assert type(key) is str
        assert val != 0.0


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_poca_optimizer_update_curiosity(
    dummy_config, curiosity_dummy_config, rnn, visual, discrete  # noqa: F811
):
    # Test evaluate
    dummy_config.reward_signals = curiosity_dummy_config
    optimizer = create_test_poca_optimizer(
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
            RewardSignalUtil.baseline_estimates_key("extrinsic"),
            RewardSignalUtil.returns_key("curiosity"),
            RewardSignalUtil.value_estimates_key("curiosity"),
            RewardSignalUtil.baseline_estimates_key("curiosity"),
        ],
    )
    # Copy memories to critic memories
    copy_buffer_fields(
        update_buffer,
        BufferKey.MEMORY,
        [BufferKey.CRITIC_MEMORY, BufferKey.BASELINE_MEMORY],
    )

    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_poca_optimizer_update_gail(gail_dummy_config, dummy_config):  # noqa: F811
    # Test evaluate
    dummy_config.reward_signals = gail_dummy_config
    config = poca_dummy_config()
    optimizer = create_test_poca_optimizer(
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
            RewardSignalUtil.baseline_estimates_key("extrinsic"),
            RewardSignalUtil.returns_key("gail"),
            RewardSignalUtil.value_estimates_key("gail"),
            RewardSignalUtil.baseline_estimates_key("gail"),
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
            RewardSignalUtil.baseline_estimates_key("extrinsic"),
            RewardSignalUtil.returns_key("gail"),
            RewardSignalUtil.value_estimates_key("gail"),
            RewardSignalUtil.baseline_estimates_key("gail"),
        ],
    )
    optimizer.update(
        update_buffer,
        num_sequences=update_buffer.num_experiences // optimizer.policy.sequence_length,
    )


def test_poca_end_episode():
    name_behavior_id = "test_brain?team=0"
    trainer = POCATrainer(
        name_behavior_id,
        10,
        TrainerSettings(max_steps=100, checkpoint_interval=10, summary_freq=20),
        True,
        False,
        0,
        "mock_model_path",
    )
    behavior_spec = BehaviorSpec(
        create_observation_specs_with_shapes([(1,)]), ActionSpec.create_discrete((2,))
    )
    parsed_behavior_id = BehaviorIdentifiers.from_name_behavior_id(name_behavior_id)
    mock_policy = trainer.create_policy(parsed_behavior_id, behavior_spec)
    trainer.add_policy(parsed_behavior_id, mock_policy)
    trajectory_queue = AgentManagerQueue("test_brain?team=0")
    policy_queue = AgentManagerQueue("test_brain?team=0")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    trainer.publish_policy_queue(policy_queue)
    time_horizon = 10
    trajectory = mb.make_fake_trajectory(
        length=time_horizon,
        observation_specs=behavior_spec.observation_specs,
        max_step_complete=False,
        action_spec=behavior_spec.action_spec,
        num_other_agents_in_group=2,
        group_reward=1.0,
        is_terminal=False,
    )
    trajectory_queue.put(trajectory)
    trainer.advance()
    # Test that some trajectoories have been injested
    for reward in trainer.collected_group_rewards.values():
        assert reward == 10
    # Test end episode
    trainer.end_episode()
    assert len(trainer.collected_group_rewards.keys()) == 0


if __name__ == "__main__":
    pytest.main()

from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
import pytest
from typing import Dict, Any
import numpy as np
import attr

# Import to avoid circular import
from mlagents.trainers.trainer.trainer_factory import TrainerFactory  # noqa F401

from mlagents.trainers.mappo.optimizer_torch import TorchMAPPOOptimizer
from mlagents.trainers.mappo.trainer import MAPPOTrainer
from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.tests import mock_brain as mb
from mlagents.trainers.tests.mock_brain import copy_buffer_fields
from mlagents.trainers.tests.test_trajectory import make_fake_trajectory
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.tests.dummy_config import (  # noqa: F401
    create_observation_specs_with_shapes,
    mappo_dummy_config,
    curiosity_dummy_config,
)
from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.settings import TrainerSettings

from mlagents_envs.base_env import ActionSpec, BehaviorSpec
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil


@pytest.fixture
def dummy_config():
    return mappo_dummy_config()


VECTOR_ACTION_SPACE = 2
VECTOR_OBS_SPACE = 8
DISCRETE_ACTION_SPACE = [3, 3, 3, 2]
BUFFER_INIT_SAMPLES = 64
NUM_AGENTS = 4

CONTINUOUS_ACTION_SPEC = ActionSpec.create_continuous(VECTOR_ACTION_SPACE)
DISCRETE_ACTION_SPEC = ActionSpec.create_discrete(tuple(DISCRETE_ACTION_SPACE))


def create_test_mappo_optimizer(dummy_config, use_rnn, use_discrete, use_visual):
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
    optimizer = TorchMAPPOOptimizer(policy, trainer_settings)
    return optimizer


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_mappo_optimizer_update(dummy_config, rnn, visual, discrete):
    optimizer = create_test_mappo_optimizer(
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
    # MAPPO has no counterfactual baseline
    assert "Losses/Baseline Loss" not in return_stats.keys()


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
def test_mappo_get_value_estimates(dummy_config, rnn, visual, discrete):
    optimizer = create_test_mappo_optimizer(
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
        value_next,
        value_memories,
    ) = optimizer.get_group_trajectory_value_estimates(
        trajectory.to_agentbuffer(),
        trajectory.next_obs,
        trajectory.next_group_obs,
        done=False,
        agent_id="test_agent",
    )
    for key, val in value_estimates.items():
        assert type(key) is str
        assert len(val) == time_horizon
    if rnn:
        assert len(value_memories) == time_horizon
        # The critic memory is kept for the next trajectory of this agent
        assert "test_agent" in optimizer.value_memory_dict
    else:
        assert value_memories is None

    (
        value_estimates,
        value_next,
        value_memories,
    ) = optimizer.get_group_trajectory_value_estimates(
        trajectory.to_agentbuffer(),
        trajectory.next_obs,
        trajectory.next_group_obs,
        done=True,
        agent_id="test_agent",
    )
    for key, val in value_next.items():
        assert type(key) is str
        assert val == 0.0
    # The episode ended, so the memory of this agent is dropped
    assert "test_agent" not in optimizer.value_memory_dict

    # Check if we ignore terminal states properly
    optimizer.reward_signals["extrinsic"]._ignore_done = True
    (
        value_estimates,
        value_next,
        value_memories,
    ) = optimizer.get_group_trajectory_value_estimates(
        trajectory.to_agentbuffer(),
        trajectory.next_obs,
        trajectory.next_group_obs,
        done=True,
    )
    for key, val in value_next.items():
        assert type(key) is str
        assert val != 0.0


def test_mappo_critic_uses_groupmate_observations(dummy_config):
    optimizer = create_test_mappo_optimizer(
        dummy_config, use_rnn=False, use_discrete=True, use_visual=False
    )
    time_horizon = 10

    def values_for(num_groupmates):
        trajectory = make_fake_trajectory(
            length=time_horizon,
            observation_specs=optimizer.policy.behavior_spec.observation_specs,
            action_spec=DISCRETE_ACTION_SPEC,
            max_step_complete=True,
            num_other_agents_in_group=num_groupmates,
        )
        values, _, _ = optimizer.get_group_trajectory_value_estimates(
            trajectory.to_agentbuffer(),
            trajectory.next_obs,
            trajectory.next_group_obs,
            done=False,
        )
        return values["extrinsic"]

    # The same agent observations, with and without groupmates: the centralized
    # critic sees a different state, so it gives a different value.
    assert not np.allclose(values_for(NUM_AGENTS), values_for(0))


@pytest.mark.parametrize("discrete", [True, False], ids=["discrete", "continuous"])
@pytest.mark.parametrize("visual", [True, False], ids=["visual", "vector"])
@pytest.mark.parametrize("rnn", [True, False], ids=["rnn", "no_rnn"])
# We need to test this separately from test_reward_signals.py to ensure no interactions
def test_mappo_optimizer_update_curiosity(
    dummy_config, curiosity_dummy_config, rnn, visual, discrete  # noqa: F811
):
    dummy_config.reward_signals = curiosity_dummy_config
    optimizer = create_test_mappo_optimizer(
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


def test_mappo_rewards_are_not_shared_between_groupmates():
    # Unlike MA-POCA, MAPPO keeps individual rewards individual (the group reward is
    # still shared), so groupmates can be given different rewards, e.g. per role.
    trainer_settings = mappo_dummy_config()
    trainer_settings.reward_signals = {
        RewardSignalType.EXTRINSIC: RewardSignalSettings(strength=1.0, gamma=0.99)
    }
    behavior_spec = mb.setup_test_behavior_specs(
        True, False, vector_action_space=[2], vector_obs_space=1
    )
    policy = TorchPolicy(
        0,
        behavior_spec,
        trainer_settings.network_settings,
        SimpleActor,
        {"conditional_sigma": False, "tanh_squash": False},
    )
    optimizer = TorchMAPPOOptimizer(policy, trainer_settings)
    assert not optimizer.reward_signals["extrinsic"].add_groupmate_rewards


def test_mappo_end_episode():
    name_behavior_id = "test_brain?team=0"
    trainer = MAPPOTrainer(
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
    # Test that some trajectories have been ingested
    for reward in trainer.collected_group_rewards.values():
        assert reward == 10
    # Test end episode
    trainer.end_episode()
    assert len(trainer.collected_group_rewards.keys()) == 0


def test_mappo_process_trajectory_sets_targets():
    name_behavior_id = "test_brain?team=0"
    trainer = MAPPOTrainer(
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
    policy = trainer.create_policy(parsed_behavior_id, behavior_spec)
    trainer.add_policy(parsed_behavior_id, policy)
    time_horizon = 10
    trajectory = mb.make_fake_trajectory(
        length=time_horizon,
        observation_specs=behavior_spec.observation_specs,
        max_step_complete=True,
        action_spec=behavior_spec.action_spec,
        num_other_agents_in_group=2,
    )
    trainer._process_trajectory(trajectory)
    buffer = trainer.update_buffer
    assert buffer.num_experiences == time_horizon
    for key in (
        BufferKey.ADVANTAGES,
        BufferKey.DISCOUNTED_RETURNS,
        RewardSignalUtil.returns_key("extrinsic"),
        RewardSignalUtil.value_estimates_key("extrinsic"),
    ):
        assert len(buffer[key]) == time_horizon
        assert np.all(np.isfinite(np.array(buffer[key], dtype=np.float32)))


if __name__ == "__main__":
    pytest.main()

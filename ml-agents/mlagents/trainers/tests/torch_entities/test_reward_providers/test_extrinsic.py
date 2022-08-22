from mlagents.trainers.buffer import BufferKey
import pytest
import numpy as np
from mlagents.trainers.torch_entities.components.reward_providers import (
    ExtrinsicRewardProvider,
    create_reward_provider,
)
from mlagents_envs.base_env import BehaviorSpec, ActionSpec
from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType
from mlagents.trainers.tests.torch_entities.test_reward_providers.utils import (
    create_agent_buffer,
)
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes


ACTIONSPEC_CONTINUOUS = ActionSpec.create_continuous(5)
ACTIONSPEC_TWODISCRETE = ActionSpec.create_discrete((2, 3))


@pytest.mark.parametrize(
    "behavior_spec",
    [
        BehaviorSpec(
            create_observation_specs_with_shapes([(10,)]), ACTIONSPEC_CONTINUOUS
        ),
        BehaviorSpec(
            create_observation_specs_with_shapes([(10,)]), ACTIONSPEC_TWODISCRETE
        ),
    ],
)
def test_construction(behavior_spec: BehaviorSpec) -> None:
    settings = RewardSignalSettings()
    settings.gamma = 0.2
    extrinsic_rp = ExtrinsicRewardProvider(behavior_spec, settings)
    assert extrinsic_rp.gamma == 0.2
    assert extrinsic_rp.name == "Extrinsic"


@pytest.mark.parametrize(
    "behavior_spec",
    [
        BehaviorSpec(
            create_observation_specs_with_shapes([(10,)]), ACTIONSPEC_CONTINUOUS
        ),
        BehaviorSpec(
            create_observation_specs_with_shapes([(10,)]), ACTIONSPEC_TWODISCRETE
        ),
    ],
)
def test_factory(behavior_spec: BehaviorSpec) -> None:
    settings = RewardSignalSettings()
    extrinsic_rp = create_reward_provider(
        RewardSignalType.EXTRINSIC, behavior_spec, settings
    )
    assert extrinsic_rp.name == "Extrinsic"


@pytest.mark.parametrize("reward", [2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "behavior_spec",
    [
        BehaviorSpec(
            create_observation_specs_with_shapes([(10,)]), ACTIONSPEC_CONTINUOUS
        ),
        BehaviorSpec(
            create_observation_specs_with_shapes([(10,)]), ACTIONSPEC_TWODISCRETE
        ),
    ],
)
def test_reward(behavior_spec: BehaviorSpec, reward: float) -> None:
    buffer = create_agent_buffer(behavior_spec, 1000, reward)
    settings = RewardSignalSettings()
    extrinsic_rp = ExtrinsicRewardProvider(behavior_spec, settings)
    generated_rewards = extrinsic_rp.evaluate(buffer)
    assert (generated_rewards == reward).all()

    # Test group rewards. Rewards should be double of the environment rewards, but shouldn't count
    # the groupmate rewards.
    buffer[BufferKey.GROUP_REWARD] = buffer[BufferKey.ENVIRONMENT_REWARDS]
    # 2 agents with identical rewards
    buffer[BufferKey.GROUPMATE_REWARDS].set(
        [np.ones(1, dtype=np.float32) * reward] * 2
        for _ in range(buffer.num_experiences)
    )
    generated_rewards = extrinsic_rp.evaluate(buffer)
    assert (generated_rewards == 2 * reward).all()

    # Test groupmate rewards. Total reward should be indiv_reward + 2 * teammate_reward + group_reward
    extrinsic_rp = ExtrinsicRewardProvider(behavior_spec, settings)
    extrinsic_rp.add_groupmate_rewards = True
    generated_rewards = extrinsic_rp.evaluate(buffer)
    assert (generated_rewards == 4 * reward).all()

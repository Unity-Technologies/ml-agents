import pytest
from mlagents.trainers.torch.components.reward_providers import (
    ExtrinsicRewardProvider,
    create_reward_provider,
)
from mlagents_envs.base_env import BehaviorSpec, ActionSpec
from mlagents.trainers.settings import RewardSignalSettings, RewardSignalType
from mlagents.trainers.tests.torch.test_reward_providers.utils import (
    create_agent_buffer,
)


ACTIONSPEC_CONTINUOUS = ActionSpec.create_continuous(5)
ACTIONSPEC_TWODISCRETE = ActionSpec.create_discrete((2, 3))


@pytest.mark.parametrize(
    "behavior_spec",
    [
        BehaviorSpec([(10,)], ACTIONSPEC_CONTINUOUS),
        BehaviorSpec([(10,)], ACTIONSPEC_TWODISCRETE),
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
        BehaviorSpec([(10,)], ACTIONSPEC_CONTINUOUS),
        BehaviorSpec([(10,)], ACTIONSPEC_TWODISCRETE),
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
        BehaviorSpec([(10,)], ACTIONSPEC_CONTINUOUS),
        BehaviorSpec([(10,)], ACTIONSPEC_TWODISCRETE),
    ],
)
def test_reward(behavior_spec: BehaviorSpec, reward: float) -> None:
    buffer = create_agent_buffer(behavior_spec, 1000, reward)
    settings = RewardSignalSettings()
    extrinsic_rp = ExtrinsicRewardProvider(behavior_spec, settings)
    generated_rewards = extrinsic_rp.evaluate(buffer)
    assert (generated_rewards == reward).all()

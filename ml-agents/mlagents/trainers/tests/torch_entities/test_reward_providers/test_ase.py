import numpy as np
import pytest
from mlagents.torch_utils import torch
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.torch_entities.components.reward_providers import (
    ASERewardProvider,
    create_reward_provider,
)
import os
from mlagents_envs.base_env import BehaviorSpec, ActionSpec
from mlagents.trainers.settings import ASESettings, RewardSignalType
from mlagents.trainers.tests.torch_entities.test_reward_providers.utils import (
    create_agent_buffer,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.components.reward_providers.ase_reward_provider import (
    DiscriminatorEncoder,
)

from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes

SEED = [42]

# currently only supports continuous actions
ACTIONSPEC_CONTINUOUS = ActionSpec.create_continuous(5)

CONTINUOUS_PATH = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    + "/test.demo"
)
DISCRETE_PATH = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    + "/testdcvis.demo"
)
SEED = [42]

# currently only supports continuous actions
ACTIONSPEC_CONTINUOUS = ActionSpec.create_continuous(2)


@pytest.mark.parametrize(
    "behavior_spec",
    [BehaviorSpec(create_observation_specs_with_shapes([(8,)]), ACTIONSPEC_CONTINUOUS)],
)
def test_construction(behavior_spec: BehaviorSpec) -> None:
    ase_settings = ASESettings(demo_path=CONTINUOUS_PATH)
    ase_rp = ASERewardProvider(behavior_spec, ase_settings)
    assert ase_rp.name == "ASE"

@pytest.mark.parametrize(
    "behavior_spec",
    [BehaviorSpec(create_observation_specs_with_shapes([(8,)]), ACTIONSPEC_CONTINUOUS)],
)
def test_factory(behavior_spec: BehaviorSpec) -> None:
    ase_settings = ASESettings(demo_path=CONTINUOUS_PATH)
    ase_rp = create_reward_provider(
        RewardSignalType.ASE, behavior_spec, ase_settings
    )
    assert ase_rp.name == "ASE"
    
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize(
    "behavior_spec",
    [
        BehaviorSpec(
            create_observation_specs_with_shapes([(8,), (24, 26, 1)]),
            ACTIONSPEC_CONTINUOUS,
        ),
    ],
)
@pytest.mark.parametrize("use_actions", [False, True])
@patch(
    "mlagents.trainers.torch_entities.components.reward_providers.ase_reward_provider.demo_to_buffer"
)

def test_reward_decreases(
    demo_to_buffer: Any, use_actions: bool, behavior_spec: BehaviorSpec, seed: int
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    buffer_expert = create_agent_buffer(behavior_spec, 1000)
    buffer_policy = create_agent_buffer(behavior_spec, 1000)
    demo_to_buffer.return_value = None, buffer_expert
    ase_settings = ASESettings(
        demo_path="", learning_rate=0.005, use_vail=False
    )
    ase_rp = create_reward_provider(
        RewardSignalType.ASE, behavior_spec, ase_settings
    )

    init_reward_expert = ase_rp.evaluate(buffer_expert)[0]
    init_reward_policy = ase_rp.evaluate(buffer_policy)[0]

    for _ in range(20):
        ase_rp.update(buffer_policy)
        reward_expert = ase_rp.evaluate(buffer_expert)[0]
        reward_policy = ase_rp.evaluate(buffer_policy)[0]
        assert reward_expert >= 0  
        assert reward_policy >= 0
    reward_expert = ase_rp.evaluate(buffer_expert)[0]
    reward_policy = ase_rp.evaluate(buffer_policy)[0]
    assert reward_expert > reward_policy  # Expert reward greater than non-expert reward
    assert (
        reward_expert > init_reward_expert
    )  # Expert reward getting better as network trains
    assert (
        reward_policy < init_reward_policy
    )  # Non-expert reward getting worse as network trains

@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize(
    "behavior_spec",
    [
        BehaviorSpec(
            create_observation_specs_with_shapes([(8,), (24, 26, 1)]),
            ACTIONSPEC_CONTINUOUS,
        ),
    ],
)
@pytest.mark.parametrize("use_actions", [False, True])
@patch(
    "mlagents.trainers.torch_entities.components.reward_providers.ase_reward_provider.demo_to_buffer"
)
def test_reward_decreases_vail(
    demo_to_buffer: Any, use_actions: bool, behavior_spec: BehaviorSpec, seed: int
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    buffer_expert = create_agent_buffer(behavior_spec, 1000)
    buffer_policy = create_agent_buffer(behavior_spec, 1000)
    demo_to_buffer.return_value = None, buffer_expert
    ase_settings = ASESettings(
        demo_path="", learning_rate=0.005, use_vail=True, use_actions=use_actions
    )
    DiscriminatorEncoder.beta = 0.0
    # we must set the initial value of beta to 0 for testing
    # If we do not, the kl-loss will dominate early and will block the estimator
    ase_rp = create_reward_provider(
        RewardSignalType.ASE, behavior_spec, ase_settings
    )

    for _ in range(20):
        ase_rp.update(buffer_policy)
        reward_expert = ase_rp.evaluate(buffer_expert)[0]
        reward_policy = ase_rp.evaluate(buffer_policy)[0]
        assert reward_expert >= 0  # GAIL / VAIL reward always positive
        assert reward_policy >= 0
    reward_expert = ase_rp.evaluate(buffer_expert)[0]
    reward_policy = ase_rp.evaluate(buffer_policy)[0]
    assert reward_expert > reward_policy  # Expert reward greater than non-expert reward

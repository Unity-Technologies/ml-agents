import pytest
import numpy as np

from mlagents_envs.base_env import (
    DecisionSteps,
    TerminalSteps,
    ActionSpec,
    BehaviorSpec,
)


def test_decision_steps():
    ds = DecisionSteps(
        obs=[np.array(range(12), dtype=np.float32).reshape(3, 4)],
        reward=np.array(range(3), dtype=np.float32),
        agent_id=np.array(range(10, 13), dtype=np.int32),
        action_mask=[np.zeros((3, 4), dtype=np.bool)],
    )

    assert ds.agent_id_to_index[10] == 0
    assert ds.agent_id_to_index[11] == 1
    assert ds.agent_id_to_index[12] == 2

    with pytest.raises(KeyError):
        assert ds.agent_id_to_index[-1] == -1

    mask_agent = ds[10].action_mask
    assert isinstance(mask_agent, list)
    assert len(mask_agent) == 1
    assert np.array_equal(mask_agent[0], np.zeros((4), dtype=np.bool))

    for agent_id in ds:
        assert ds.agent_id_to_index[agent_id] in range(3)


def test_empty_decision_steps():
    specs = BehaviorSpec(
        observation_shapes=[(3, 2), (5,)], action_spec=ActionSpec.create_continuous(3)
    )
    ds = DecisionSteps.empty(specs)
    assert len(ds.obs) == 2
    assert ds.obs[0].shape == (0, 3, 2)
    assert ds.obs[1].shape == (0, 5)


def test_terminal_steps():
    ts = TerminalSteps(
        obs=[np.array(range(12), dtype=np.float32).reshape(3, 4)],
        reward=np.array(range(3), dtype=np.float32),
        agent_id=np.array(range(10, 13), dtype=np.int32),
        interrupted=np.array([1, 0, 1], dtype=np.bool),
    )

    assert ts.agent_id_to_index[10] == 0
    assert ts.agent_id_to_index[11] == 1
    assert ts.agent_id_to_index[12] == 2

    assert ts[10].interrupted
    assert not ts[11].interrupted
    assert ts[12].interrupted

    with pytest.raises(KeyError):
        assert ts.agent_id_to_index[-1] == -1

    for agent_id in ts:
        assert ts.agent_id_to_index[agent_id] in range(3)


def test_empty_terminal_steps():
    specs = BehaviorSpec(
        observation_shapes=[(3, 2), (5,)], action_spec=ActionSpec.create_continuous(3)
    )
    ts = TerminalSteps.empty(specs)
    assert len(ts.obs) == 2
    assert ts.obs[0].shape == (0, 3, 2)
    assert ts.obs[1].shape == (0, 5)


def test_specs():
    specs = ActionSpec.create_continuous(3)
    assert specs.discrete_branches == ()
    assert specs.discrete_size == 0
    assert specs.continuous_size == 3
    assert specs.empty_action(5).continuous.shape == (5, 3)
    assert specs.empty_action(5).continuous.dtype == np.float32

    specs = ActionSpec.create_discrete((3,))
    assert specs.discrete_branches == (3,)
    assert specs.discrete_size == 1
    assert specs.continuous_size == 0
    assert specs.empty_action(5).discrete.shape == (5, 1)
    assert specs.empty_action(5).discrete.dtype == np.int32

    specs = ActionSpec(3, (3,))
    assert specs.continuous_size == 3
    assert specs.discrete_branches == (3,)
    assert specs.discrete_size == 1
    assert specs.empty_action(5).continuous.shape == (5, 3)
    assert specs.empty_action(5).continuous.dtype == np.float32
    assert specs.empty_action(5).discrete.shape == (5, 1)
    assert specs.empty_action(5).discrete.dtype == np.int32


def test_action_generator():
    # Continuous
    action_len = 30
    specs = ActionSpec.create_continuous(action_len)
    zero_action = specs.empty_action(4).continuous
    assert np.array_equal(zero_action, np.zeros((4, action_len), dtype=np.float32))
    print(specs.random_action(4))
    random_action = specs.random_action(4).continuous
    print(random_action)
    assert random_action.dtype == np.float32
    assert random_action.shape == (4, action_len)
    assert np.min(random_action) >= -1
    assert np.max(random_action) <= 1

    # Discrete
    action_shape = (10, 20, 30)
    specs = ActionSpec.create_discrete(action_shape)
    zero_action = specs.empty_action(4).discrete
    assert np.array_equal(zero_action, np.zeros((4, len(action_shape)), dtype=np.int32))

    random_action = specs.random_action(4).discrete
    assert random_action.dtype == np.int32
    assert random_action.shape == (4, len(action_shape))
    assert np.min(random_action) >= 0
    for index, branch_size in enumerate(action_shape):
        assert np.max(random_action[:, index]) < branch_size

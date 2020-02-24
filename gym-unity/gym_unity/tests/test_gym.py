from unittest import mock
import pytest
import numpy as np

from gym import spaces
from gym_unity.envs import (
    UnityEnv,
    UnityGymException,
    AgentIdIndexMapper,
    AgentIdIndexMapperSlow,
)
from mlagents_envs.base_env import AgentGroupSpec, ActionType, BatchedStepResult


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_gym_wrapper(mock_env):
    mock_spec = create_mock_group_spec()
    mock_step = create_mock_vector_step_result()
    setup_mock_unityenvironment(mock_env, mock_spec, mock_step)

    env = UnityEnv(" ", use_visual=False, multiagent=False)
    assert isinstance(env, UnityEnv)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert env.observation_space.contains(obs)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_multi_agent(mock_env):
    mock_spec = create_mock_group_spec()
    mock_step = create_mock_vector_step_result(num_agents=2)
    setup_mock_unityenvironment(mock_env, mock_spec, mock_step)

    with pytest.raises(UnityGymException):
        UnityEnv(" ", multiagent=False)

    env = UnityEnv(" ", use_visual=False, multiagent=True)
    assert isinstance(env.reset(), list)
    actions = [env.action_space.sample() for i in range(env.number_agents)]
    obs, rew, done, info = env.step(actions)
    assert isinstance(obs, list)
    assert isinstance(rew, list)
    assert isinstance(done, list)
    assert isinstance(info, dict)


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_branched_flatten(mock_env):
    mock_spec = create_mock_group_spec(
        vector_action_space_type="discrete", vector_action_space_size=[2, 2, 3]
    )
    mock_step = create_mock_vector_step_result(num_agents=1)
    setup_mock_unityenvironment(mock_env, mock_spec, mock_step)

    env = UnityEnv(" ", use_visual=False, multiagent=False, flatten_branched=True)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 12
    assert env._flattener.lookup_action(0) == [0, 0, 0]
    assert env._flattener.lookup_action(11) == [1, 1, 2]

    # Check that False produces a MultiDiscrete
    env = UnityEnv(" ", use_visual=False, multiagent=False, flatten_branched=False)
    assert isinstance(env.action_space, spaces.MultiDiscrete)


@pytest.mark.parametrize("use_uint8", [True, False], ids=["float", "uint8"])
@mock.patch("gym_unity.envs.UnityEnvironment")
def test_gym_wrapper_visual(mock_env, use_uint8):
    mock_spec = create_mock_group_spec(number_visual_observations=1)
    mock_step = create_mock_vector_step_result(number_visual_observations=1)
    setup_mock_unityenvironment(mock_env, mock_spec, mock_step)

    env = UnityEnv(" ", use_visual=True, multiagent=False, uint8_visual=use_uint8)
    assert isinstance(env, UnityEnv)
    assert isinstance(env.reset(), np.ndarray)
    actions = env.action_space.sample()
    assert actions.shape[0] == 2
    obs, rew, done, info = env.step(actions)
    assert env.observation_space.contains(obs)
    assert isinstance(obs, np.ndarray)
    assert isinstance(rew, float)
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_sanitize_action_shuffled_id(mock_env):
    mock_spec = create_mock_group_spec(
        vector_action_space_type="discrete", vector_action_space_size=[2, 2, 3]
    )
    mock_step = create_mock_vector_step_result(num_agents=5)
    mock_step.agent_id = np.array(range(5))
    setup_mock_unityenvironment(mock_env, mock_spec, mock_step)
    env = UnityEnv(" ", use_visual=False, multiagent=True)

    shuffled_step_result = create_mock_vector_step_result(num_agents=5)
    shuffled_order = [4, 2, 3, 1, 0]
    shuffled_step_result.reward = np.array(shuffled_order)
    shuffled_step_result.agent_id = np.array(shuffled_order)
    sanitized_result = env._sanitize_info(shuffled_step_result)
    for expected_reward, reward in zip(range(5), sanitized_result.reward):
        assert expected_reward == reward
    for expected_agent_id, agent_id in zip(range(5), sanitized_result.agent_id):
        assert expected_agent_id == agent_id


@mock.patch("gym_unity.envs.UnityEnvironment")
def test_sanitize_action_one_agent_done(mock_env):
    mock_spec = create_mock_group_spec(
        vector_action_space_type="discrete", vector_action_space_size=[2, 2, 3]
    )
    mock_step = create_mock_vector_step_result(num_agents=5)
    mock_step.agent_id = np.array(range(5))
    setup_mock_unityenvironment(mock_env, mock_spec, mock_step)
    env = UnityEnv(" ", use_visual=False, multiagent=True)

    received_step_result = create_mock_vector_step_result(num_agents=6)
    received_step_result.agent_id = np.array(range(6))
    # agent #3 (id = 2) is Done
    received_step_result.done = np.array([False] * 2 + [True] + [False] * 3)
    sanitized_result = env._sanitize_info(received_step_result)
    for expected_agent_id, agent_id in zip([0, 1, 5, 3, 4], sanitized_result.agent_id):
        assert expected_agent_id == agent_id


# Helper methods


def create_mock_group_spec(
    number_visual_observations=0,
    vector_action_space_type="continuous",
    vector_observation_space_size=3,
    vector_action_space_size=None,
):
    """
    Creates a mock BrainParameters object with parameters.
    """
    # Avoid using mutable object as default param
    act_type = ActionType.DISCRETE
    if vector_action_space_type == "continuous":
        act_type = ActionType.CONTINUOUS
        if vector_action_space_size is None:
            vector_action_space_size = 2
        else:
            vector_action_space_size = vector_action_space_size[0]
    else:
        if vector_action_space_size is None:
            vector_action_space_size = (2,)
        else:
            vector_action_space_size = tuple(vector_action_space_size)
    obs_shapes = [(vector_observation_space_size,)]
    for _ in range(number_visual_observations):
        obs_shapes += [(8, 8, 3)]
    return AgentGroupSpec(obs_shapes, act_type, vector_action_space_size)


def create_mock_vector_step_result(num_agents=1, number_visual_observations=0):
    """
    Creates a mock BatchedStepResult with vector observations. Imitates constant
    vector observations, rewards, dones, and agents.

    :int num_agents: Number of "agents" to imitate in your BatchedStepResult values.
    """
    obs = [np.array([num_agents * [1, 2, 3]]).reshape(num_agents, 3)]
    if number_visual_observations:
        obs += [np.zeros(shape=(num_agents, 8, 8, 3), dtype=np.float32)]
    rewards = np.array(num_agents * [1.0])
    done = np.array(num_agents * [False])
    agents = np.array(range(0, num_agents))
    return BatchedStepResult(obs, rewards, done, done, agents, None)


def setup_mock_unityenvironment(mock_env, mock_spec, mock_result):
    """
    Takes a mock UnityEnvironment and adds the appropriate properties, defined by the mock
    GroupSpec and BatchedStepResult.

    :Mock mock_env: A mock UnityEnvironment, usually empty.
    :Mock mock_spec: An AgentGroupSpec object that specifies the params of this environment.
    :Mock mock_result: A BatchedStepResult object that will be returned at each step and reset.
    """
    mock_env.return_value.get_agent_groups.return_value = ["MockBrain"]
    mock_env.return_value.get_agent_group_spec.return_value = mock_spec
    mock_env.return_value.get_step_result.return_value = mock_result


@pytest.mark.parametrize("mapper_cls", [AgentIdIndexMapper, AgentIdIndexMapperSlow])
def test_agent_id_index_mapper(mapper_cls):
    mapper = mapper_cls()
    initial_agent_ids = [1001, 1002, 1003, 1004]
    mapper.set_initial_agents(initial_agent_ids)

    # Mark some agents as done with their last rewards.
    mapper.mark_agent_done(1001, 42.0)
    mapper.mark_agent_done(1004, 1337.0)

    # Now add new agents, and get the rewards of the agent they replaced.
    old_reward1 = mapper.register_new_agent_id(2001)
    old_reward2 = mapper.register_new_agent_id(2002)

    # Order of the rewards don't matter
    assert {old_reward1, old_reward2} == {42.0, 1337.0}

    new_agent_ids = [1002, 1003, 2001, 2002]
    permutation = mapper.get_id_permutation(new_agent_ids)
    # Make sure it's actually a permutation - needs to contain 0..N-1 with no repeats.
    assert set(permutation) == set(range(0, 4))

    # For initial agents that were in the initial group, they need to be in the same slot.
    # Agents that were added later can appear in any free slot.
    permuted_ids = [new_agent_ids[i] for i in permutation]
    for idx, agent_id in enumerate(initial_agent_ids):
        if agent_id in permuted_ids:
            assert permuted_ids[idx] == agent_id

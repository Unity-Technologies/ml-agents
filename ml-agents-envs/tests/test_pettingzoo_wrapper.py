import numpy as np
import pytest

from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.envs.env_helpers import _unwrap_batch_steps
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from simple_test_envs import SimpleEnvironment, MultiAgentEnvironment
from pettingzoo.test import api_test, parallel_api_test

NUM_TEST_CYCLES = 100


def _make_terminal_batch(interrupted):
    """A batch with a single terminating agent and no live (decision) agents."""
    obs = [np.array([[1, 2, 3]], dtype=np.float32)]
    reward = np.array([1.0])
    interrupted_arr = np.array([interrupted], dtype=bool)
    agent_id = np.array([0])
    group_id = np.array([0])
    group_reward = np.array([0.0])
    decision_batch = DecisionSteps([], [], [], [], [], [])
    termination_batch = TerminalSteps(
        obs, reward, interrupted_arr, agent_id, group_id, group_reward
    )
    return decision_batch, termination_batch


@pytest.mark.parametrize(
    "interrupted,exp_terminated,exp_truncated",
    [(True, False, True), (False, True, False)],
    ids=["truncated", "terminated"],
)
def test_unwrap_batch_steps_terminated_truncated(
    interrupted, exp_terminated, exp_truncated
):
    # An interrupted terminal step is a truncation; a non-interrupted one is a
    # termination. This is the pettingzoo-side counterpart of the gym split.
    (_, _, terminations, truncations, _, _, infos, _) = _unwrap_batch_steps(
        _make_terminal_batch(interrupted), "MockBrain"
    )
    agent_id = "MockBrain?agent_id=0"
    assert terminations[agent_id] is exp_terminated
    assert truncations[agent_id] is exp_truncated
    assert infos[agent_id]["interrupted"] is exp_truncated


def test_single_agent_aec():
    unity_env = SimpleEnvironment(["test_single"])
    env = UnityAECEnv(unity_env)
    api_test(env, num_cycles=NUM_TEST_CYCLES, verbose_progress=False)


def test_multi_agent_aec():
    unity_env = MultiAgentEnvironment(["test_multi_1", "test_multi_2"], num_agents=2)
    env = UnityAECEnv(unity_env)
    api_test(env, num_cycles=NUM_TEST_CYCLES, verbose_progress=False)


def test_single_agent_parallel():
    unity_env = SimpleEnvironment(["test_single"])
    env = UnityParallelEnv(unity_env)
    parallel_api_test(env, num_cycles=NUM_TEST_CYCLES)


def test_multi_agent_parallel():
    unity_env = MultiAgentEnvironment(
        ["test_multi_1", "test_multi_2", "test_multi_3"], num_agents=3
    )
    env = UnityParallelEnv(unity_env)
    parallel_api_test(env, num_cycles=NUM_TEST_CYCLES)


def test_reset_seed_reseeds_action_spaces():
    # reset(seed=...) on a long-lived wrapper must actually reseed the existing
    # action spaces so sampling is reproducible, not just store self._seed.
    unity_env = SimpleEnvironment(["test_single"])
    env = UnityAECEnv(unity_env)

    env.reset(seed=1337)
    agent = env.possible_agents[0]
    first = env.action_space(agent).sample()
    env.reset(seed=1337)
    second = env.action_space(agent).sample()

    assert np.array_equal(first, second)

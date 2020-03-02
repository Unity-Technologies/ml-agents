from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec
from mlagents.trainers.action_info import ActionInfo
from unittest.mock import MagicMock
import numpy as np


def basic_mock_brain():
    mock_brain = MagicMock()
    mock_brain.vector_action_space_type = "continuous"
    mock_brain.vector_observation_space_size = 1
    mock_brain.vector_action_space_size = [1]
    return mock_brain


def basic_params():
    return {"use_recurrent": False, "model_path": "my/path"}


class FakePolicy(TFPolicy):
    def create_tf_graph(self):
        pass

    def get_trainable_variables(self):
        return []


def test_take_action_returns_empty_with_no_agents():
    test_seed = 3
    policy = FakePolicy(test_seed, basic_mock_brain(), basic_params())
    # Doesn't really matter what this is
    dummy_groupspec = AgentGroupSpec([(1,)], "continuous", 1)
    no_agent_step = BatchedStepResult.empty(dummy_groupspec)
    result = policy.get_action(no_agent_step)
    assert result == ActionInfo.empty()


def test_take_action_returns_nones_on_missing_values():
    test_seed = 3
    policy = FakePolicy(test_seed, basic_mock_brain(), basic_params())
    policy.evaluate = MagicMock(return_value={})
    policy.save_memories = MagicMock()
    step_with_agents = BatchedStepResult(
        [],
        np.array([], dtype=np.float32),
        np.array([False], dtype=np.bool),
        np.array([], dtype=np.bool),
        np.array([0]),
        None,
    )
    result = policy.get_action(step_with_agents, worker_id=0)
    assert result == ActionInfo(None, None, {}, [0])


def test_take_action_returns_action_info_when_available():
    test_seed = 3
    policy = FakePolicy(test_seed, basic_mock_brain(), basic_params())
    policy_eval_out = {
        "action": np.array([1.0], dtype=np.float32),
        "memory_out": np.array([[2.5]], dtype=np.float32),
        "value": np.array([1.1], dtype=np.float32),
    }
    policy.evaluate = MagicMock(return_value=policy_eval_out)
    step_with_agents = BatchedStepResult(
        [],
        np.array([], dtype=np.float32),
        np.array([False], dtype=np.bool),
        np.array([], dtype=np.bool),
        np.array([0]),
        None,
    )
    result = policy.get_action(step_with_agents)
    expected = ActionInfo(
        policy_eval_out["action"], policy_eval_out["value"], policy_eval_out, [0]
    )
    assert result == expected

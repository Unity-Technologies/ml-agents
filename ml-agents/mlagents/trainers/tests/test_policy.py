from mlagents.trainers.tf_policy import *
from mlagents.envs.brain import AgentInfo
from unittest.mock import MagicMock


def basic_mock_brain():
    mock_brain = MagicMock()
    mock_brain.vector_action_space_type = "continuous"
    return mock_brain


def basic_params():
    return {"use_recurrent": False, "model_path": "my/path"}


def test_get_action_returns_empty_with_no_agents():
    test_seed = 3
    policy = TFPolicy(test_seed, basic_mock_brain(), basic_params())
    empty_agent_infos = []
    result = policy.get_action(empty_agent_infos)
    assert result == ActionInfo([], [], [], None, None)


def test_get_action_returns_nones_on_missing_values():
    test_seed = 3
    policy = TFPolicy(test_seed, basic_mock_brain(), basic_params())
    policy.evaluate = MagicMock(return_value={})
    agent_infos = [AgentInfo("TestBrain", [], [], None, None, None, "an-agent-id")]
    result = policy.get_action(agent_infos)
    assert result == ActionInfo(None, None, None, None, {})


def test_get_action_returns_action_info_when_available():
    test_seed = 3
    policy = TFPolicy(test_seed, basic_mock_brain(), basic_params())
    policy_eval_out = {
        "action": np.array([1.0]),
        "memory_out": np.array([2.5]),
        "value": np.array([1.1]),
    }
    policy.evaluate = MagicMock(return_value=policy_eval_out)
    agent_infos = [AgentInfo("TestBrain", [], [], None, None, None, "an-agent-id")]
    result = policy.get_action(agent_infos)
    expected = ActionInfo(
        policy_eval_out["action"],
        policy_eval_out["memory_out"],
        None,
        policy_eval_out["value"],
        policy_eval_out,
    )
    assert result == expected

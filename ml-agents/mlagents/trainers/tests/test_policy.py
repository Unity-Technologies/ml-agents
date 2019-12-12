from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.brain import BrainInfo
from mlagents.trainers.action_info import ActionInfo
from unittest.mock import MagicMock
import numpy as np


def basic_mock_brain():
    mock_brain = MagicMock()
    mock_brain.vector_action_space_type = "continuous"
    return mock_brain


def basic_params():
    return {"use_recurrent": False, "model_path": "my/path"}


def test_take_action_returns_empty_with_no_agents():
    test_seed = 3
    policy = TFPolicy(test_seed, basic_mock_brain(), basic_params())
    no_agent_brain_info = BrainInfo([], [], [], agents=[])
    result = policy.get_action(no_agent_brain_info)
    assert result == ActionInfo([], [], {})


def test_take_action_returns_nones_on_missing_values():
    test_seed = 3
    policy = TFPolicy(test_seed, basic_mock_brain(), basic_params())
    policy.evaluate = MagicMock(return_value={})
    policy.save_memories = MagicMock()
    brain_info_with_agents = BrainInfo(
        [], [], [], agents=["an-agent-id"], local_done=[False]
    )
    result = policy.get_action(brain_info_with_agents)
    assert result == ActionInfo(None, None, {})


def test_take_action_returns_action_info_when_available():
    test_seed = 3
    policy = TFPolicy(test_seed, basic_mock_brain(), basic_params())
    policy_eval_out = {
        "action": np.array([1.0], dtype=np.float32),
        "memory_out": np.array([[2.5]], dtype=np.float32),
        "value": np.array([1.1], dtype=np.float32),
    }
    policy.evaluate = MagicMock(return_value=policy_eval_out)
    brain_info_with_agents = BrainInfo(
        [], [], [], agents=["an-agent-id"], local_done=[False]
    )
    result = policy.get_action(brain_info_with_agents)
    expected = ActionInfo(
        policy_eval_out["action"], policy_eval_out["value"], policy_eval_out
    )
    assert result == expected

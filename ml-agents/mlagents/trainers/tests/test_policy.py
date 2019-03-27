from mlagents.trainers.policy import *
from unittest.mock import MagicMock

def basic_mock_brain():
    mock_brain = MagicMock()
    mock_brain.vector_action_space_type = "continuous"
    return mock_brain

def basic_params():
    return {
        "use_recurrent": False,
        "model_path": "my/path"
    }


def test_take_action_returns_empty_with_no_agents():
    test_seed = 3
    policy = Policy(test_seed, basic_mock_brain(), basic_params())
    no_agent_brain_info = BrainInfo([], [], [], agents=[])
    result = policy.get_action(no_agent_brain_info)
    assert(result == ActionInfo([], [], [], None, None))


def test_take_action_returns_nones_on_missing_values():
    test_seed = 3
    policy = Policy(test_seed, basic_mock_brain(), basic_params())
    policy.evaluate = MagicMock(return_value={})
    brain_info_with_agents = BrainInfo([], [], [], agents=['an-agent-id'])
    result = policy.get_action(brain_info_with_agents)
    assert(result == ActionInfo(None, None, None, None, {}))


def test_take_action_returns_action_info_when_available():
    test_seed = 3
    policy = Policy(test_seed, basic_mock_brain(), basic_params())
    policy_eval_out = {
        'action': np.array([1.0]),
        'memory_out': np.array([2.5]),
        'value': np.array([1.1])
    }
    policy.evaluate = MagicMock(return_value=policy_eval_out)
    brain_info_with_agents = BrainInfo([], [], [], agents=['an-agent-id'])
    result = policy.get_action(brain_info_with_agents)
    expected = ActionInfo(
        policy_eval_out['action'],
        policy_eval_out['memory_out'],
        None,
        policy_eval_out['value'],
        policy_eval_out
    )
    assert (result == expected)

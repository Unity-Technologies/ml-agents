from unittest import mock
import pytest
import mlagents.trainers.tests.mock_brain as mb
import numpy as np
from mlagents.trainers.agent_processor import AgentProcessor
from mlagents.trainers.stats import StatsReporter


def create_mock_brain():
    mock_brain = mb.create_mock_brainparams(
        vector_action_space_type="continuous",
        vector_action_space_size=[2],
        vector_observation_space_size=8,
        number_visual_observations=1,
    )
    return mock_brain


def create_mock_policy():
    mock_policy = mock.Mock()
    mock_policy.reward_signals = {}
    mock_policy.retrieve_memories.return_value = np.zeros((1, 1), dtype=np.float32)
    mock_policy.retrieve_previous_action.return_value = np.zeros(
        (1, 1), dtype=np.float32
    )
    return mock_policy


@pytest.mark.parametrize("num_vis_obs", [0, 1, 2], ids=["vec", "1 viz", "2 viz"])
def test_agentprocessor(num_vis_obs):
    policy = create_mock_policy()
    trainer = mock.Mock()
    name_behavior_id = "test_brain_name"
    processor = AgentProcessor(
        trainer,
        policy,
        name_behavior_id,
        max_trajectory_length=5,
        stats_reporter=StatsReporter("testcat"),
    )
    fake_action_outputs = {
        "action": [0.1, 0.1],
        "entropy": np.array([1.0], dtype=np.float32),
        "learning_rate": 1.0,
        "pre_action": [0.1, 0.1],
        "log_probs": [0.1, 0.1],
    }
    mock_braininfo = mb.create_mock_braininfo(
        num_agents=2,
        num_vector_observations=8,
        num_vector_acts=2,
        num_vis_observations=num_vis_obs,
    )
    for _ in range(5):
        processor.add_experiences(mock_braininfo, mock_braininfo, fake_action_outputs)

    # Assert that two trajectories have been added to the Trainer
    assert len(trainer.process_trajectory.call_args_list) == 2

    # Assert that the trajectory is of length 5
    trajectory = trainer.process_trajectory.call_args_list[0][0][0]
    assert len(trajectory.steps) == 5

    # Assert that the AgentProcessor is empty
    assert len(processor.experience_buffers[0]) == 0

from unittest import mock
import pytest
import mlagents.trainers.tests.mock_brain as mb
import numpy as np
from mlagents.trainers.agent_processor import (
    AgentProcessor,
    AgentManager,
    AgentManagerQueue,
)
from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.brain_conversion_utils import get_global_agent_id


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
    tqueue = mock.Mock()
    name_behavior_id = "test_brain_name"
    processor = AgentProcessor(
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
    mock_step = mb.create_mock_batchedstep(
        num_agents=2,
        num_vector_observations=8,
        action_shape=[2],
        num_vis_observations=num_vis_obs,
    )
    fake_action_info = ActionInfo(
        action=[0.1, 0.1],
        value=[0.1, 0.1],
        outputs=fake_action_outputs,
        agent_ids=mock_step.agent_id,
    )
    processor.publish_trajectory_queue(tqueue)
    # This is like the initial state after the env reset
    processor.add_experiences(mock_step, 0, ActionInfo.empty())
    for _ in range(5):
        processor.add_experiences(mock_step, 0, fake_action_info)

    # Assert that two trajectories have been added to the Trainer
    assert len(tqueue.put.call_args_list) == 2

    # Assert that the trajectory is of length 5
    trajectory = tqueue.put.call_args_list[0][0][0]
    assert len(trajectory.steps) == 5

    # Assert that the AgentProcessor is empty
    assert len(processor.experience_buffers[0]) == 0

    # Test empty BatchedStepResult
    mock_step = mb.create_mock_batchedstep(
        num_agents=0,
        num_vector_observations=8,
        action_shape=[2],
        num_vis_observations=num_vis_obs,
    )
    processor.add_experiences(mock_step, 0, ActionInfo([], [], {}, []))
    # Assert that the AgentProcessor is still empty
    assert len(processor.experience_buffers[0]) == 0


def test_agent_deletion():
    policy = create_mock_policy()
    tqueue = mock.Mock()
    name_behavior_id = "test_brain_name"
    processor = AgentProcessor(
        policy,
        name_behavior_id,
        max_trajectory_length=5,
        stats_reporter=StatsReporter("testcat"),
    )

    fake_action_outputs = {
        "action": [0.1],
        "entropy": np.array([1.0], dtype=np.float32),
        "learning_rate": 1.0,
        "pre_action": [0.1],
        "log_probs": [0.1],
    }
    mock_step = mb.create_mock_batchedstep(
        num_agents=1,
        num_vector_observations=8,
        action_shape=[2],
        num_vis_observations=0,
    )
    mock_done_step = mb.create_mock_batchedstep(
        num_agents=1,
        num_vector_observations=8,
        action_shape=[2],
        num_vis_observations=0,
        done=True,
    )
    fake_action_info = ActionInfo(
        action=[0.1],
        value=[0.1],
        outputs=fake_action_outputs,
        agent_ids=mock_step.agent_id,
    )

    processor.publish_trajectory_queue(tqueue)
    # This is like the initial state after the env reset
    processor.add_experiences(mock_step, 0, ActionInfo.empty())

    # Run 3 trajectories, with different workers (to simulate different agents)
    add_calls = []
    remove_calls = []
    for _ep in range(3):
        for _ in range(5):
            processor.add_experiences(mock_step, _ep, fake_action_info)
            add_calls.append(mock.call([get_global_agent_id(_ep, 0)], [0.1]))
        processor.add_experiences(mock_done_step, _ep, fake_action_info)
        # Make sure we don't add experiences from the prior agents after the done
        remove_calls.append(mock.call([get_global_agent_id(_ep, 0)]))

    policy.save_previous_action.assert_has_calls(add_calls)
    policy.remove_previous_action.assert_has_calls(remove_calls)
    # Check that there are no experiences left
    assert len(processor.experience_buffers.keys()) == 0
    assert len(processor.last_take_action_outputs.keys()) == 0
    assert len(processor.episode_steps.keys()) == 0
    assert len(processor.episode_rewards.keys()) == 0


def test_end_episode():
    policy = create_mock_policy()
    tqueue = mock.Mock()
    name_behavior_id = "test_brain_name"
    processor = AgentProcessor(
        policy,
        name_behavior_id,
        max_trajectory_length=5,
        stats_reporter=StatsReporter("testcat"),
    )

    fake_action_outputs = {
        "action": [0.1],
        "entropy": np.array([1.0], dtype=np.float32),
        "learning_rate": 1.0,
        "pre_action": [0.1],
        "log_probs": [0.1],
    }
    mock_step = mb.create_mock_batchedstep(
        num_agents=1,
        num_vector_observations=8,
        action_shape=[2],
        num_vis_observations=0,
    )
    fake_action_info = ActionInfo(
        action=[0.1],
        value=[0.1],
        outputs=fake_action_outputs,
        agent_ids=mock_step.agent_id,
    )

    processor.publish_trajectory_queue(tqueue)
    # This is like the initial state after the env reset
    processor.add_experiences(mock_step, 0, ActionInfo.empty())
    # Run 3 trajectories, with different workers (to simulate different agents)
    remove_calls = []
    for _ep in range(3):
        remove_calls.append(mock.call([get_global_agent_id(_ep, 0)]))
        for _ in range(5):
            processor.add_experiences(mock_step, _ep, fake_action_info)
            # Make sure we don't add experiences from the prior agents after the done

    # Call end episode
    processor.end_episode()
    # Check that we removed every agent
    policy.remove_previous_action.assert_has_calls(remove_calls)
    # Check that there are no experiences left
    assert len(processor.experience_buffers.keys()) == 0
    assert len(processor.last_take_action_outputs.keys()) == 0
    assert len(processor.episode_steps.keys()) == 0
    assert len(processor.episode_rewards.keys()) == 0


def test_agent_manager():
    policy = create_mock_policy()
    name_behavior_id = "test_brain_name"
    manager = AgentManager(
        policy,
        name_behavior_id,
        max_trajectory_length=5,
        stats_reporter=StatsReporter("testcat"),
    )
    assert len(manager.trajectory_queues) == 1
    assert isinstance(manager.trajectory_queues[0], AgentManagerQueue)


def test_agent_manager_queue():
    queue = AgentManagerQueue(behavior_id="testbehavior")
    trajectory = mock.Mock(spec=Trajectory)
    assert queue.empty()
    queue.put(trajectory)
    assert not queue.empty()
    queue_traj = queue.get_nowait()
    assert isinstance(queue_traj, Trajectory)
    assert queue.empty()

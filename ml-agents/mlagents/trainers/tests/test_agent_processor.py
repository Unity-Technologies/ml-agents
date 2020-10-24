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
from mlagents.trainers.stats import StatsReporter, StatsSummary
from mlagents.trainers.behavior_id_utils import get_global_agent_id
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod

from mlagents_envs.base_env import ActionSpec


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
    mock_decision_steps, mock_terminal_steps = mb.create_mock_steps(
        num_agents=2,
        observation_shapes=[(8,)] + num_vis_obs * [(84, 84, 3)],
        action_spec=ActionSpec.create_continuous(2),
    )
    fake_action_info = ActionInfo(
        action=[0.1, 0.1],
        value=[0.1, 0.1],
        outputs=fake_action_outputs,
        agent_ids=mock_decision_steps.agent_id,
    )
    processor.publish_trajectory_queue(tqueue)
    # This is like the initial state after the env reset
    processor.add_experiences(
        mock_decision_steps, mock_terminal_steps, 0, ActionInfo.empty()
    )
    for _ in range(5):
        processor.add_experiences(
            mock_decision_steps, mock_terminal_steps, 0, fake_action_info
        )

    # Assert that two trajectories have been added to the Trainer
    assert len(tqueue.put.call_args_list) == 2

    # Assert that the trajectory is of length 5
    trajectory = tqueue.put.call_args_list[0][0][0]
    assert len(trajectory.steps) == 5

    # Assert that the AgentProcessor is empty
    assert len(processor.experience_buffers[0]) == 0

    # Test empty steps
    mock_decision_steps, mock_terminal_steps = mb.create_mock_steps(
        num_agents=0,
        observation_shapes=[(8,)] + num_vis_obs * [(84, 84, 3)],
        action_spec=ActionSpec.create_continuous(2),
    )
    processor.add_experiences(
        mock_decision_steps, mock_terminal_steps, 0, ActionInfo([], [], {}, [])
    )
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
    mock_decision_step, mock_terminal_step = mb.create_mock_steps(
        num_agents=1,
        observation_shapes=[(8,)],
        action_spec=ActionSpec.create_continuous(2),
    )
    mock_done_decision_step, mock_done_terminal_step = mb.create_mock_steps(
        num_agents=1,
        observation_shapes=[(8,)],
        action_spec=ActionSpec.create_continuous(2),
        done=True,
    )
    fake_action_info = ActionInfo(
        action=[0.1],
        value=[0.1],
        outputs=fake_action_outputs,
        agent_ids=mock_decision_step.agent_id,
    )

    processor.publish_trajectory_queue(tqueue)
    # This is like the initial state after the env reset
    processor.add_experiences(
        mock_decision_step, mock_terminal_step, 0, ActionInfo.empty()
    )

    # Run 3 trajectories, with different workers (to simulate different agents)
    add_calls = []
    remove_calls = []
    for _ep in range(3):
        for _ in range(5):
            processor.add_experiences(
                mock_decision_step, mock_terminal_step, _ep, fake_action_info
            )
            add_calls.append(mock.call([get_global_agent_id(_ep, 0)], [0.1]))
        processor.add_experiences(
            mock_done_decision_step, mock_done_terminal_step, _ep, fake_action_info
        )
        # Make sure we don't add experiences from the prior agents after the done
        remove_calls.append(mock.call([get_global_agent_id(_ep, 0)]))

    policy.save_previous_action.assert_has_calls(add_calls)
    policy.remove_previous_action.assert_has_calls(remove_calls)
    # Check that there are no experiences left
    assert len(processor.experience_buffers.keys()) == 0
    assert len(processor.last_take_action_outputs.keys()) == 0
    assert len(processor.episode_steps.keys()) == 0
    assert len(processor.episode_rewards.keys()) == 0
    assert len(processor.last_step_result.keys()) == 0

    # check that steps with immediate dones don't add to dicts
    processor.add_experiences(
        mock_done_decision_step, mock_done_terminal_step, 0, ActionInfo.empty()
    )
    assert len(processor.experience_buffers.keys()) == 0
    assert len(processor.last_take_action_outputs.keys()) == 0
    assert len(processor.episode_steps.keys()) == 0
    assert len(processor.episode_rewards.keys()) == 0
    assert len(processor.last_step_result.keys()) == 0


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
    mock_decision_step, mock_terminal_step = mb.create_mock_steps(
        num_agents=1,
        observation_shapes=[(8,)],
        action_spec=ActionSpec.create_continuous(2),
    )
    fake_action_info = ActionInfo(
        action=[0.1],
        value=[0.1],
        outputs=fake_action_outputs,
        agent_ids=mock_decision_step.agent_id,
    )

    processor.publish_trajectory_queue(tqueue)
    # This is like the initial state after the env reset
    processor.add_experiences(
        mock_decision_step, mock_terminal_step, 0, ActionInfo.empty()
    )
    # Run 3 trajectories, with different workers (to simulate different agents)
    remove_calls = []
    for _ep in range(3):
        remove_calls.append(mock.call([get_global_agent_id(_ep, 0)]))
        for _ in range(5):
            processor.add_experiences(
                mock_decision_step, mock_terminal_step, _ep, fake_action_info
            )
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


def test_agent_manager_stats():
    policy = mock.Mock()
    stats_reporter = StatsReporter("FakeCategory")
    writer = mock.Mock()
    stats_reporter.add_writer(writer)
    manager = AgentManager(policy, "MyBehavior", stats_reporter)

    all_env_stats = [
        {
            "averaged": [(1.0, StatsAggregationMethod.AVERAGE)],
            "most_recent": [(2.0, StatsAggregationMethod.MOST_RECENT)],
        },
        {
            "averaged": [(3.0, StatsAggregationMethod.AVERAGE)],
            "most_recent": [(4.0, StatsAggregationMethod.MOST_RECENT)],
        },
    ]
    for env_stats in all_env_stats:
        manager.record_environment_stats(env_stats, worker_id=0)

    expected_stats = {
        "averaged": StatsSummary(mean=2.0, std=mock.ANY, num=2),
        "most_recent": StatsSummary(mean=4.0, std=0.0, num=1),
    }
    stats_reporter.write_stats(123)
    writer.write_stats.assert_any_call("FakeCategory", expected_stats, 123)

    # clean up our Mock from the global list
    StatsReporter.writers.remove(writer)

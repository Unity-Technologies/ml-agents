import os
import unittest
from unittest import mock
import pytest
import mlagents.trainers.tests.mock_brain as mb
from mlagents.trainers.policy.checkpoint_manager import ModelCheckpoint
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.tests.test_buffer import construct_fake_buffer
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.policy import Policy
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes
from mlagents_envs.base_env import ActionSpec
import os.path


# Add concrete implementations of abstract methods
class FakeTrainer(RLTrainer):
    def set_is_policy_updating(self, is_updating):
        self.update_policy = is_updating

    def get_policy(self, name_behavior_id):
        return mock.Mock()

    def _is_ready_update(self):
        return True

    def _update_policy(self):
        return self.update_policy

    def add_policy(self, mock_behavior_id, mock_policy):
        def checkpoint_path(brain_name, step):
            onnx_file_path = os.path.join(
                self.model_saver.model_path, f"{brain_name}-{step}.onnx"
            )
            other_file_paths = [
                os.path.join(self.model_saver.model_path, f"{brain_name}-{step}.pt")
            ]
            return onnx_file_path, other_file_paths

        self.policies[mock_behavior_id] = mock_policy
        mock_model_saver = mock.Mock()
        mock_model_saver.model_path = self.artifact_path
        mock_model_saver.save_checkpoint.side_effect = checkpoint_path
        self.model_saver = mock_model_saver

    def create_optimizer(self) -> TorchOptimizer:
        return mock.Mock()

    def create_policy(
        self,
        parsed_behavior_id: BehaviorIdentifiers,
        behavior_spec: BehaviorSpec,
    ) -> Policy:
        return mock.Mock()

    def _process_trajectory(self, trajectory):
        super()._process_trajectory(trajectory)


def create_rl_trainer():
    trainer = FakeTrainer(
        "test_trainer",
        TrainerSettings(max_steps=100, checkpoint_interval=10, summary_freq=20),
        True,
        False,
        "mock_model_path",
        0,
    )
    trainer.set_is_policy_updating(True)
    return trainer


def test_rl_trainer():
    trainer = create_rl_trainer()
    agent_id = "0"
    trainer.collected_rewards["extrinsic"] = {agent_id: 3}
    # Test end episode
    trainer.end_episode()
    for rewards in trainer.collected_rewards.values():
        for agent_id in rewards:
            assert rewards[agent_id] == 0


def test_clear_update_buffer():
    trainer = create_rl_trainer()
    trainer.update_buffer = construct_fake_buffer(0)
    trainer._clear_update_buffer()
    for _, arr in trainer.update_buffer.items():
        assert len(arr) == 0


@mock.patch("mlagents.trainers.trainer.trainer.Trainer.save_model")
def test_advance(mocked_save_model):
    trainer = create_rl_trainer()
    mock_policy = mock.Mock()
    trainer.add_policy("TestBrain", mock_policy)
    trajectory_queue = AgentManagerQueue("testbrain")
    policy_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    trainer.publish_policy_queue(policy_queue)
    time_horizon = 10
    trajectory = mb.make_fake_trajectory(
        length=time_horizon,
        observation_specs=create_observation_specs_with_shapes([(1,)]),
        max_step_complete=True,
        action_spec=ActionSpec.create_discrete((2,)),
    )
    trajectory_queue.put(trajectory)

    trainer.advance()
    policy_queue.get_nowait()
    # Check that get_step is correct
    assert trainer.get_step == time_horizon
    # Check that we can turn off the trainer and that the buffer is cleared
    for _ in range(0, 5):
        trajectory_queue.put(trajectory)
        trainer.advance()
        # Check that there is stuff in the policy queue
        policy_queue.get_nowait()

    # Check that if the policy doesn't update, we don't push it to the queue
    trainer.set_is_policy_updating(False)
    for _ in range(0, 10):
        trajectory_queue.put(trajectory)
        trainer.advance()
        # Check that there nothing  in the policy queue
        with pytest.raises(AgentManagerQueue.Empty):
            policy_queue.get_nowait()

    # Check that no model has been saved
    assert not trainer.should_still_train
    assert mocked_save_model.call_count == 0


@mock.patch("mlagents.trainers.trainer.trainer.StatsReporter.write_stats")
@mock.patch(
    "mlagents.trainers.trainer.rl_trainer.ModelCheckpointManager.add_checkpoint"
)
def test_summary_checkpoint(mock_add_checkpoint, mock_write_summary):
    trainer = create_rl_trainer()
    mock_policy = mock.Mock()
    trainer.add_policy("TestBrain", mock_policy)
    trajectory_queue = AgentManagerQueue("testbrain")
    policy_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    trainer.publish_policy_queue(policy_queue)
    time_horizon = 10
    summary_freq = trainer.trainer_settings.summary_freq
    checkpoint_interval = trainer.trainer_settings.checkpoint_interval
    trajectory = mb.make_fake_trajectory(
        length=time_horizon,
        observation_specs=create_observation_specs_with_shapes([(1,)]),
        max_step_complete=True,
        action_spec=ActionSpec.create_discrete((2,)),
    )
    # Check that we can turn off the trainer and that the buffer is cleared
    num_trajectories = 5
    for _ in range(0, num_trajectories):
        trajectory_queue.put(trajectory)
        trainer.advance()
        # Check that there is stuff in the policy queue
        policy_queue.get_nowait()

    # Check that we have called write_summary the appropriate number of times
    calls = [
        mock.call(step)
        for step in range(summary_freq, num_trajectories * time_horizon, summary_freq)
    ]
    mock_write_summary.assert_has_calls(calls, any_order=True)

    checkpoint_range = range(
        checkpoint_interval, num_trajectories * time_horizon, checkpoint_interval
    )
    calls = [mock.call(trainer.brain_name, step) for step in checkpoint_range]

    trainer.model_saver.save_checkpoint.assert_has_calls(calls, any_order=True)
    export_ext = "onnx"

    add_checkpoint_calls = [
        mock.call(
            trainer.brain_name,
            ModelCheckpoint(
                step,
                f"{trainer.model_saver.model_path}{os.path.sep}{trainer.brain_name}-{step}.{export_ext}",
                None,
                mock.ANY,
                [
                    f"{trainer.model_saver.model_path}{os.path.sep}{trainer.brain_name}-{step}.pt"
                ],
            ),
            trainer.trainer_settings.keep_checkpoints,
        )
        for step in checkpoint_range
    ]
    mock_add_checkpoint.assert_has_calls(add_checkpoint_calls)


def test_update_buffer_append():
    trainer = create_rl_trainer()
    mock_policy = mock.Mock()
    trainer.add_policy("TestBrain", mock_policy)
    trajectory_queue = AgentManagerQueue("testbrain")
    policy_queue = AgentManagerQueue("testbrain")
    trainer.subscribe_trajectory_queue(trajectory_queue)
    trainer.publish_policy_queue(policy_queue)
    time_horizon = 10
    trajectory = mb.make_fake_trajectory(
        length=time_horizon,
        observation_specs=create_observation_specs_with_shapes([(1,)]),
        max_step_complete=True,
        action_spec=ActionSpec.create_discrete((2,)),
    )
    agentbuffer_trajectory = trajectory.to_agentbuffer()
    assert trainer.update_buffer.num_experiences == 0

    # Check that if we append, our update buffer gets longer.
    # max_steps = 100
    for i in range(10):
        trainer._process_trajectory(trajectory)
        trainer._append_to_update_buffer(agentbuffer_trajectory)
        assert trainer.update_buffer.num_experiences == (i + 1) * time_horizon

    # Check that if we append after stopping training, nothing happens.
    # We process enough trajectories to hit max steps
    trainer.set_is_policy_updating(False)
    trainer._process_trajectory(trajectory)
    trainer._append_to_update_buffer(agentbuffer_trajectory)
    assert trainer.update_buffer.num_experiences == (i + 1) * time_horizon


class RLTrainerWarningTest(unittest.TestCase):
    def test_warning_group_reward(self):
        with self.assertLogs("mlagents.trainers", level="WARN") as cm:
            rl_trainer = create_rl_trainer()
            # This one should warn
            trajectory = mb.make_fake_trajectory(
                length=10,
                observation_specs=create_observation_specs_with_shapes([(1,)]),
                max_step_complete=True,
                action_spec=ActionSpec.create_discrete((2,)),
                group_reward=1.0,
            )
            buff = trajectory.to_agentbuffer()
            rl_trainer._warn_if_group_reward(buff)
            assert len(cm.output) > 0
            len_of_first_warning = len(cm.output)

            rl_trainer = create_rl_trainer()
            # This one shouldn't
            trajectory = mb.make_fake_trajectory(
                length=10,
                observation_specs=create_observation_specs_with_shapes([(1,)]),
                max_step_complete=True,
                action_spec=ActionSpec.create_discrete((2,)),
            )
            buff = trajectory.to_agentbuffer()
            rl_trainer._warn_if_group_reward(buff)
            # Make sure warnings don't get bigger
            assert len(cm.output) == len_of_first_warning

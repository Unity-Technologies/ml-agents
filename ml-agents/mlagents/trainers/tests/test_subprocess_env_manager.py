from unittest import mock
from unittest.mock import Mock, MagicMock, call, ANY
import unittest
import pytest
from queue import Empty as EmptyQueue

from mlagents.trainers.settings import RunOptions
from mlagents.trainers.subprocess_env_manager import (
    SubprocessEnvManager,
    EnvironmentResponse,
    StepResponse,
    EnvironmentCommand,
)
from mlagents.trainers.env_manager import EnvironmentStep
from mlagents_envs.base_env import BaseEnv
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
)
from mlagents.trainers.tests.simple_test_envs import (
    SimpleEnvironment,
    UnexpectedExceptionEnvironment,
)
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.tests.check_env_trains import (
    check_environment_trains,
    DebugWriter,
)
from mlagents.trainers.tests.dummy_config import ppo_dummy_config


def mock_env_factory(worker_id):
    return mock.create_autospec(spec=BaseEnv)


class MockEnvWorker:
    def __init__(self, worker_id, resp=None):
        self.worker_id = worker_id
        self.process = None
        self.conn = None
        self.send = Mock()
        self.recv = Mock(return_value=resp)
        self.waiting = False


def create_worker_mock(worker_id, step_queue, env_factor, engine_c):
    return MockEnvWorker(
        worker_id, EnvironmentResponse(EnvironmentCommand.RESET, worker_id, worker_id)
    )


class SubprocessEnvManagerTest(unittest.TestCase):
    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_environments_are_created(self, mock_create_worker):
        mock_create_worker.side_effect = create_worker_mock
        run_options = RunOptions()
        env = SubprocessEnvManager(mock_env_factory, run_options, 2)
        # Creates two processes
        env.create_worker.assert_has_calls(
            [
                mock.call(0, env.step_queue, mock_env_factory, run_options),
                mock.call(1, env.step_queue, mock_env_factory, run_options),
            ]
        )
        self.assertEqual(len(env.env_workers), 2)

    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_reset_passes_reset_params(self, mock_create_worker):
        mock_create_worker.side_effect = create_worker_mock
        manager = SubprocessEnvManager(mock_env_factory, RunOptions(), 1)
        params = {"test": "params"}
        manager._reset_env(params)
        manager.env_workers[0].send.assert_called_with(
            EnvironmentCommand.RESET, (params)
        )

    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_reset_collects_results_from_all_envs(self, mock_create_worker):
        mock_create_worker.side_effect = create_worker_mock
        manager = SubprocessEnvManager(mock_env_factory, RunOptions(), 4)

        params = {"test": "params"}
        res = manager._reset_env(params)
        for i, env in enumerate(manager.env_workers):
            env.send.assert_called_with(EnvironmentCommand.RESET, (params))
            env.recv.assert_called()
            # Check that the "last steps" are set to the value returned for each step
            self.assertEqual(
                manager.env_workers[i].previous_step.current_all_step_result, i
            )
        assert res == list(map(lambda ew: ew.previous_step, manager.env_workers))

    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_training_behaviors_collects_results_from_all_envs(
        self, mock_create_worker
    ):
        def create_worker_mock(worker_id, step_queue, env_factor, engine_c):
            return MockEnvWorker(
                worker_id,
                EnvironmentResponse(
                    EnvironmentCommand.RESET, worker_id, {f"key{worker_id}": worker_id}
                ),
            )

        mock_create_worker.side_effect = create_worker_mock
        manager = SubprocessEnvManager(mock_env_factory, RunOptions(), 4)

        res = manager.training_behaviors
        for env in manager.env_workers:
            env.send.assert_called_with(EnvironmentCommand.BEHAVIOR_SPECS)
            env.recv.assert_called()
        for worker_id in range(4):
            assert f"key{worker_id}" in res
            assert res[f"key{worker_id}"] == worker_id

    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_step_takes_steps_for_all_non_waiting_envs(self, mock_create_worker):
        mock_create_worker.side_effect = create_worker_mock
        manager = SubprocessEnvManager(mock_env_factory, RunOptions(), 3)
        manager.step_queue = Mock()
        manager.step_queue.get_nowait.side_effect = [
            EnvironmentResponse(EnvironmentCommand.STEP, 0, StepResponse(0, None, {})),
            EnvironmentResponse(EnvironmentCommand.STEP, 1, StepResponse(1, None, {})),
            EmptyQueue(),
        ]
        step_mock = Mock()
        last_steps = [Mock(), Mock(), Mock()]
        manager.env_workers[0].previous_step = last_steps[0]
        manager.env_workers[1].previous_step = last_steps[1]
        manager.env_workers[2].previous_step = last_steps[2]
        manager.env_workers[2].waiting = True
        manager._take_step = Mock(return_value=step_mock)
        res = manager._step()
        for i, env in enumerate(manager.env_workers):
            if i < 2:
                env.send.assert_called_with(EnvironmentCommand.STEP, step_mock)
                manager.step_queue.get_nowait.assert_called()
                # Check that the "last steps" are set to the value returned for each step
                self.assertEqual(
                    manager.env_workers[i].previous_step.current_all_step_result, i
                )
        assert res == [
            manager.env_workers[0].previous_step,
            manager.env_workers[1].previous_step,
        ]

    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_crashed_env_restarts(self, mock_create_worker):
        crashing_worker = MockEnvWorker(
            0, EnvironmentResponse(EnvironmentCommand.RESET, 0, 0)
        )
        restarting_worker = MockEnvWorker(
            0, EnvironmentResponse(EnvironmentCommand.RESET, 0, 0)
        )
        healthy_worker = MockEnvWorker(
            1, EnvironmentResponse(EnvironmentCommand.RESET, 1, 1)
        )
        mock_create_worker.side_effect = [
            crashing_worker,
            healthy_worker,
            restarting_worker,
        ]
        manager = SubprocessEnvManager(mock_env_factory, RunOptions(), 2)
        manager.step_queue = Mock()
        manager.step_queue.get_nowait.side_effect = [
            EnvironmentResponse(
                EnvironmentCommand.ENV_EXITED,
                0,
                UnityCommunicationException("Test msg"),
            ),
            EnvironmentResponse(EnvironmentCommand.CLOSED, 0, None),
            EnvironmentResponse(EnvironmentCommand.STEP, 1, StepResponse(0, None, {})),
            EmptyQueue(),
            EnvironmentResponse(EnvironmentCommand.STEP, 0, StepResponse(1, None, {})),
            EnvironmentResponse(EnvironmentCommand.STEP, 1, StepResponse(2, None, {})),
            EmptyQueue(),
        ]
        step_mock = Mock()
        last_steps = [Mock(), Mock(), Mock()]
        assert crashing_worker is manager.env_workers[0]
        assert healthy_worker is manager.env_workers[1]
        crashing_worker.previous_step = last_steps[0]
        crashing_worker.waiting = True
        healthy_worker.previous_step = last_steps[1]
        healthy_worker.waiting = True
        manager._take_step = Mock(return_value=step_mock)
        manager._step()
        healthy_worker.send.assert_has_calls(
            [
                call(EnvironmentCommand.ENVIRONMENT_PARAMETERS, ANY),
                call(EnvironmentCommand.RESET, ANY),
                call(EnvironmentCommand.STEP, ANY),
            ]
        )
        restarting_worker.send.assert_has_calls(
            [
                call(EnvironmentCommand.ENVIRONMENT_PARAMETERS, ANY),
                call(EnvironmentCommand.RESET, ANY),
                call(EnvironmentCommand.STEP, ANY),
            ]
        )

    @mock.patch("mlagents.trainers.subprocess_env_manager.SubprocessEnvManager._step")
    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.training_behaviors",
        new_callable=mock.PropertyMock,
    )
    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.create_worker"
    )
    def test_advance(self, mock_create_worker, training_behaviors_mock, step_mock):
        brain_name = "testbrain"
        action_info_dict = {brain_name: MagicMock()}
        mock_create_worker.side_effect = create_worker_mock
        env_manager = SubprocessEnvManager(mock_env_factory, RunOptions(), 3)
        training_behaviors_mock.return_value = [brain_name]
        agent_manager_mock = mock.Mock()
        mock_policy = mock.Mock()
        agent_manager_mock.policy_queue.get_nowait.side_effect = [
            mock_policy,
            mock_policy,
            AgentManagerQueue.Empty(),
        ]
        env_manager.set_agent_manager(brain_name, agent_manager_mock)

        step_info_dict = {brain_name: (Mock(), Mock())}
        env_stats = {
            "averaged": (1.0, StatsAggregationMethod.AVERAGE),
            "most_recent": (2.0, StatsAggregationMethod.MOST_RECENT),
        }
        step_info = EnvironmentStep(step_info_dict, 0, action_info_dict, env_stats)
        step_mock.return_value = [step_info]
        env_manager.process_steps(env_manager.get_steps())

        # Test add_experiences
        env_manager._step.assert_called_once()

        agent_manager_mock.add_experiences.assert_called_once_with(
            step_info.current_all_step_result[brain_name][0],
            step_info.current_all_step_result[brain_name][1],
            0,
            step_info.brain_name_to_action_info[brain_name],
        )

        # Test policy queue
        assert env_manager.policies[brain_name] == mock_policy
        assert agent_manager_mock.policy == mock_policy


@pytest.mark.parametrize("num_envs", [1, 4])
def test_subprocess_env_endtoend(num_envs):
    def simple_env_factory(worker_id, config):
        env = SimpleEnvironment(["1D"], action_sizes=(0, 1))
        return env

    env_manager = SubprocessEnvManager(simple_env_factory, RunOptions(), num_envs)
    # Run PPO using env_manager
    check_environment_trains(
        simple_env_factory(0, []),
        {"1D": ppo_dummy_config()},
        env_manager=env_manager,
        success_threshold=None,
    )
    # Note we can't check the env's rewards directly (since they're in separate processes) so we
    # check the StatsReporter's debug stat writer's last reward.
    assert isinstance(StatsReporter.writers[0], DebugWriter)
    assert all(
        val > 0.7 for val in StatsReporter.writers[0].get_last_rewards().values()
    )
    env_manager.close()


class CustomTestOnlyException(Exception):
    pass


@pytest.mark.parametrize("num_envs", [1, 4])
def test_subprocess_failing_step(num_envs):
    def failing_step_env_factory(_worker_id, _config):
        env = UnexpectedExceptionEnvironment(
            ["1D"], use_discrete=True, to_raise=CustomTestOnlyException
        )
        return env

    env_manager = SubprocessEnvManager(failing_step_env_factory, RunOptions())
    # Expect the exception raised to be routed back up to the top level.
    with pytest.raises(CustomTestOnlyException):
        check_environment_trains(
            failing_step_env_factory(0, []),
            {"1D": ppo_dummy_config()},
            env_manager=env_manager,
            success_threshold=None,
        )
    env_manager.close()


@pytest.mark.parametrize("num_envs", [1, 4])
def test_subprocess_env_raises_errors(num_envs):
    def failing_env_factory(worker_id, config):
        import time

        # Sleep momentarily to allow time for the EnvManager to be waiting for the
        # subprocess response.  We won't be able to capture failures from the subprocess
        # that cause it to close the pipe before we can send the first message.
        time.sleep(0.5)
        raise UnityEnvironmentException()

    env_manager = SubprocessEnvManager(failing_env_factory, RunOptions(), num_envs)
    with pytest.raises(UnityEnvironmentException):
        env_manager.reset()
    env_manager.close()

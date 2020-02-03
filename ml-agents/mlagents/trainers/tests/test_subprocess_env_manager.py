from unittest import mock
from unittest.mock import Mock, MagicMock
import unittest
from queue import Empty as EmptyQueue

from mlagents.trainers.subprocess_env_manager import (
    SubprocessEnvManager,
    EnvironmentResponse,
    StepResponse,
)
from mlagents.trainers.env_manager import EnvironmentStep
from mlagents_envs.base_env import BaseEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig


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


class SubprocessEnvManagerTest(unittest.TestCase):
    def test_environments_are_created(self):
        SubprocessEnvManager.create_worker = MagicMock()
        env = SubprocessEnvManager(mock_env_factory, EngineConfig.default_config(), 2)
        # Creates two processes
        env.create_worker.assert_has_calls(
            [
                mock.call(
                    0, env.step_queue, mock_env_factory, EngineConfig.default_config()
                ),
                mock.call(
                    1, env.step_queue, mock_env_factory, EngineConfig.default_config()
                ),
            ]
        )
        self.assertEqual(len(env.env_workers), 2)

    def test_reset_passes_reset_params(self):
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory, engine_c: MockEnvWorker(
            worker_id, EnvironmentResponse("reset", worker_id, worker_id)
        )
        manager = SubprocessEnvManager(
            mock_env_factory, EngineConfig.default_config(), 1
        )
        params = {"test": "params"}
        manager._reset_env(params)
        manager.env_workers[0].send.assert_called_with("reset", (params))

    def test_reset_collects_results_from_all_envs(self):
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory, engine_c: MockEnvWorker(
            worker_id, EnvironmentResponse("reset", worker_id, worker_id)
        )
        manager = SubprocessEnvManager(
            mock_env_factory, EngineConfig.default_config(), 4
        )

        params = {"test": "params"}
        res = manager._reset_env(params)
        for i, env in enumerate(manager.env_workers):
            env.send.assert_called_with("reset", (params))
            env.recv.assert_called()
            # Check that the "last steps" are set to the value returned for each step
            self.assertEqual(
                manager.env_workers[i].previous_step.current_all_step_result, i
            )
        assert res == list(map(lambda ew: ew.previous_step, manager.env_workers))

    def test_step_takes_steps_for_all_non_waiting_envs(self):
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory, engine_c: MockEnvWorker(
            worker_id, EnvironmentResponse("step", worker_id, worker_id)
        )
        manager = SubprocessEnvManager(
            mock_env_factory, EngineConfig.default_config(), 3
        )
        manager.step_queue = Mock()
        manager.step_queue.get_nowait.side_effect = [
            EnvironmentResponse("step", 0, StepResponse(0, None)),
            EnvironmentResponse("step", 1, StepResponse(1, None)),
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
                env.send.assert_called_with("step", step_mock)
                manager.step_queue.get_nowait.assert_called()
                # Check that the "last steps" are set to the value returned for each step
                self.assertEqual(
                    manager.env_workers[i].previous_step.current_all_step_result, i
                )
        assert res == [
            manager.env_workers[0].previous_step,
            manager.env_workers[1].previous_step,
        ]

    @mock.patch("mlagents.trainers.subprocess_env_manager.SubprocessEnvManager._step")
    @mock.patch(
        "mlagents.trainers.subprocess_env_manager.SubprocessEnvManager.external_brains",
        new_callable=mock.PropertyMock,
    )
    def test_advance(self, external_brains_mock, step_mock):
        brain_name = "testbrain"
        action_info_dict = {brain_name: MagicMock()}
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory, engine_c: MockEnvWorker(
            worker_id, EnvironmentResponse("step", worker_id, worker_id)
        )
        env_manager = SubprocessEnvManager(
            mock_env_factory, EngineConfig.default_config(), 3
        )
        external_brains_mock.return_value = [brain_name]
        agent_manager_mock = mock.Mock()
        env_manager.set_agent_manager(brain_name, agent_manager_mock)

        step_info_dict = {brain_name: Mock()}
        step_info = EnvironmentStep(step_info_dict, 0, action_info_dict)
        step_mock.return_value = [step_info]
        env_manager.advance()

        # Test add_experiences
        env_manager._step.assert_called_once()

        agent_manager_mock.add_experiences.assert_called_once_with(
            step_info.current_all_step_result[brain_name],
            0,
            step_info.brain_name_to_action_info[brain_name],
        )

        # Test policy queue
        mock_policy = mock.Mock()
        agent_manager_mock.policy_queue.get_nowait.return_value = mock_policy
        env_manager.advance()
        assert env_manager.policies[brain_name] == mock_policy
        assert agent_manager_mock.policy == mock_policy

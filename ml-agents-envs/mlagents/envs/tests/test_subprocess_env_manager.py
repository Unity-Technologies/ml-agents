import unittest.mock as mock
from unittest.mock import Mock, MagicMock
import unittest
import cloudpickle
from queue import Empty as EmptyQueue

from mlagents.envs.subprocess_env_manager import (
    SubprocessEnvManager,
    EnvironmentResponse,
    EnvironmentCommand,
    worker,
    StepResponse,
)
from mlagents.envs.env_manager import AgentStep
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.brain import AgentInfo


def mock_env_factory(worker_id: int):
    return mock.create_autospec(spec=BaseUnityEnvironment)


class MockEnvWorker:
    def __init__(self, worker_id, resp=None):
        self.worker_id = worker_id
        self.previous_agent_steps = {}
        self.process = None
        self.conn = None
        self.send = Mock()
        self.recv = Mock(return_value=resp)
        self.waiting = False


class SubprocessEnvManagerTest(unittest.TestCase):
    def test_environments_are_created(self):
        SubprocessEnvManager.create_worker = MagicMock()
        env = SubprocessEnvManager(mock_env_factory, 2)
        # Creates two processes
        env.create_worker.assert_has_calls(
            [
                mock.call(0, env.step_queue, mock_env_factory),
                mock.call(1, env.step_queue, mock_env_factory),
            ]
        )
        self.assertEqual(len(env.env_workers), 2)

    def test_worker_step_resets_on_global_done(self):
        env_mock = Mock()
        env_mock.reset = Mock(return_value="reset_data")
        env_mock.global_done = True

        def mock_global_done_env_factory(worker_id: int):
            return env_mock

        mock_parent_connection = Mock()
        mock_step_queue = Mock()
        step_command = EnvironmentCommand("step", (None, None, None, None))
        close_command = EnvironmentCommand("close")
        mock_parent_connection.recv.side_effect = [step_command, close_command]
        mock_parent_connection.send = Mock()

        worker(
            mock_parent_connection,
            mock_step_queue,
            cloudpickle.dumps(mock_global_done_env_factory),
            0,
        )

        # recv called twice to get step and close command
        self.assertEqual(mock_parent_connection.recv.call_count, 2)

        expected_step_response = StepResponse(
            all_agent_info="reset_data", timer_root=mock.ANY
        )

        # worker returns the data from the reset
        mock_step_queue.put.assert_called_with(
            EnvironmentResponse("step", 0, expected_step_response)
        )

    def test_reset_passes_reset_params(self):
        agent_info = AgentInfo("TestBrain", None, None, None, None, None, "agent-1")
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory: MockEnvWorker(
            worker_id, EnvironmentResponse("reset", worker_id, [agent_info])
        )
        manager = SubprocessEnvManager(mock_env_factory, 1)
        params = {"test": "params"}
        manager.reset(params, False)
        manager.env_workers[0].send.assert_called_with("reset", (params, False, None))

    def test_reset_collects_results_from_all_envs(self):
        agent_info = AgentInfo("TestBrain", None, None, None, None, None, "agent-1")
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory: MockEnvWorker(
            worker_id, EnvironmentResponse("reset", worker_id, [agent_info])
        )
        n_env = 4
        manager = SubprocessEnvManager(mock_env_factory, n_env=n_env)

        params = {"test": "params"}
        res = manager.reset(params)
        for i, env in enumerate(manager.env_workers):
            env.send.assert_called_with("reset", (params, True, None))
            env.recv.assert_called()
            # Check that the "last steps" are set to the value returned for each step
            self.assertEqual(
                manager.env_workers[i]
                .previous_agent_steps["agent-1"]
                .current_agent_info,
                agent_info,
            )
        assert res == [AgentStep(None, agent_info, None)] * n_env

    def test_step_takes_steps_for_all_non_waiting_envs(self):
        SubprocessEnvManager.create_worker = lambda em, worker_id, step_queue, env_factory: MockEnvWorker(
            worker_id, EnvironmentResponse("step", worker_id, worker_id)
        )
        manager = SubprocessEnvManager(mock_env_factory, 3)
        manager.step_queue = Mock()
        agent_infos = [
            AgentInfo("TestBrain", None, None, None, None, None, "agent-1"),
            AgentInfo("TestBrain", None, None, None, None, None, "agent-2"),
        ]
        manager.step_queue.get_nowait.side_effect = [
            EnvironmentResponse("step", 0, StepResponse(agent_infos, None)),
            EnvironmentResponse("step", 1, StepResponse(agent_infos, None)),
            EmptyQueue(),
        ]
        step_mock = Mock()
        last_steps = [Mock(), Mock(), Mock()]
        manager.env_workers[0].previous_agent_steps["agent-1"] = last_steps[0]
        manager.env_workers[0].previous_agent_steps["agent-2"] = last_steps[0]
        manager.env_workers[1].previous_agent_steps["agent-1"] = last_steps[1]
        manager.env_workers[1].previous_agent_steps["agent-2"] = last_steps[1]
        # manager.env_workers[2].previous_agent_steps['agent-1'] = last_steps[2]
        manager.env_workers[2].waiting = True
        manager._take_step = Mock(return_value=step_mock)
        manager.step()
        for i, env in enumerate(manager.env_workers):
            if i < 2:
                env.send.assert_called_with("step", step_mock)
                manager.step_queue.get_nowait.assert_called()
                # Check that the "last steps" are set to the value returned for each step
                self.assertEqual(
                    manager.env_workers[i]
                    .previous_agent_steps["agent-1"]
                    .current_agent_info,
                    agent_infos[0],
                )
                self.assertEqual(
                    manager.env_workers[i]
                    .previous_agent_steps["agent-1"]
                    .previous_agent_info,
                    last_steps[i].current_agent_info,
                )
                self.assertEqual(
                    manager.env_workers[i]
                    .previous_agent_steps["agent-2"]
                    .current_agent_info,
                    agent_infos[1],
                )
        assert list(manager.env_workers[0].previous_agent_steps.values()) + list(
            manager.env_workers[1].previous_agent_steps.values()
        )

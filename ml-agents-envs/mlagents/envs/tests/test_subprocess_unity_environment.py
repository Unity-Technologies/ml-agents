import unittest.mock as mock
from unittest.mock import MagicMock
import unittest

from mlagents.envs.subprocess_environment import *
from mlagents.envs import UnityEnvironmentException, BrainInfo


def mock_env_factory(worker_id: int):
    return mock.create_autospec(spec=BaseUnityEnvironment)


class MockEnvWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.process = None
        self.conn = None
        self.send = MagicMock()
        self.recv = MagicMock()


class SubprocessEnvironmentTest(unittest.TestCase):
    def test_environments_are_created(self):
        SubprocessUnityEnvironment.create_worker = MagicMock()
        env = SubprocessUnityEnvironment(mock_env_factory, 2)
        # Creates two processes
        self.assertEqual(env.create_worker.call_args_list, [
            mock.call(0, mock_env_factory),
            mock.call(1, mock_env_factory)
        ])
        self.assertEqual(len(env.envs), 2)

    def test_step_async_fails_when_waiting(self):
        env = SubprocessUnityEnvironment(mock_env_factory, 0)
        env.waiting = True
        with self.assertRaises(UnityEnvironmentException):
            env.step_async(vector_action=[])

    @staticmethod
    def test_step_async_splits_input_by_agent_count():
        env = SubprocessUnityEnvironment(mock_env_factory, 0)
        env.env_agent_counts = {
            'MockBrain': [1, 3, 5]
        }
        env.envs = [
            MockEnvWorker(0),
            MockEnvWorker(1),
            MockEnvWorker(2),
        ]
        env_0_actions = [[1.0, 2.0]]
        env_1_actions = ([[3.0, 4.0]] * 3)
        env_2_actions = ([[5.0, 6.0]] * 5)
        vector_action = {
            'MockBrain': env_0_actions + env_1_actions + env_2_actions
        }
        env.step_async(vector_action=vector_action)
        env.envs[0].send.assert_called_with('step', ({'MockBrain': env_0_actions}, {}, {}, {}))
        env.envs[1].send.assert_called_with('step', ({'MockBrain': env_1_actions}, {}, {}, {}))
        env.envs[2].send.assert_called_with('step', ({'MockBrain': env_2_actions}, {}, {}, {}))

    def test_step_async_sets_waiting(self):
        env = SubprocessUnityEnvironment(mock_env_factory, 0)
        env.step_async(vector_action=[])
        self.assertTrue(env.waiting)

    def test_step_await_fails_if_not_waiting(self):
        env = SubprocessUnityEnvironment(mock_env_factory, 0)
        with self.assertRaises(UnityEnvironmentException):
            env.step_await()

    def test_step_await_combines_brain_info(self):
        all_brain_info_env0 = {
            'MockBrain': BrainInfo([], [[1.0, 2.0], [1.0, 2.0]], [], agents=[1, 2], memory=np.zeros((0,0)))
        }
        all_brain_info_env1 = {
            'MockBrain': BrainInfo([], [[3.0, 4.0]], [], agents=[3], memory=np.zeros((0,0)))
        }
        env_worker_0 = MockEnvWorker(0)
        env_worker_0.recv.return_value = EnvironmentResponse('step', 0, all_brain_info_env0)
        env_worker_1 = MockEnvWorker(1)
        env_worker_1.recv.return_value = EnvironmentResponse('step', 1, all_brain_info_env1)
        env = SubprocessUnityEnvironment(mock_env_factory, 0)
        env.envs = [env_worker_0, env_worker_1]
        env.waiting = True
        combined_braininfo = env.step_await()['MockBrain']
        self.assertEqual(
            combined_braininfo.vector_observations.tolist(),
            [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]
        )
        self.assertEqual(combined_braininfo.agents, ['0-1', '0-2', '1-3'])

from typing import *
import cloudpickle

from mlagents.envs import UnityEnvironment
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager
from mlagents.envs import AllBrainInfo, BrainParameters


class EnvironmentCommand(NamedTuple):
    name: str
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    name: str
    worker_id: int
    payload: Any


class UnityEnvWorker(NamedTuple):
    process: Process
    worker_id: int
    conn: Connection

    def send(self, name: str, payload=None):
        try:
            cmd = EnvironmentCommand(name, payload)
            self.conn.send(cmd)
        except (BrokenPipeError, EOFError):
            raise KeyboardInterrupt

    def recv(self) -> EnvironmentResponse:
        try:
            response: EnvironmentResponse = self.conn.recv()
            return response
        except (BrokenPipeError, EOFError):
            raise KeyboardInterrupt

    def close(self):
        try:
            self.conn.send(EnvironmentCommand("close"))
        except (BrokenPipeError, EOFError):
            pass
        self.process.join()


def worker(parent_conn: Connection, pickled_env_factory: str, worker_id: int):
    env_factory: Callable[[int], UnityEnvironment] = cloudpickle.loads(
        pickled_env_factory
    )
    env = env_factory(worker_id)

    def _send_response(cmd_name, payload):
        parent_conn.send(EnvironmentResponse(cmd_name, worker_id, payload))

    try:
        while True:
            cmd: EnvironmentCommand = parent_conn.recv()
            if cmd.name == "step":
                vector_action, memory, text_action, value = cmd.payload
                if env.global_done:
                    all_brain_info = env.reset()
                else:
                    all_brain_info = env.step(vector_action, memory, text_action, value)
                _send_response("step", all_brain_info)
            elif cmd.name == "external_brains":
                _send_response("external_brains", env.external_brains)
            elif cmd.name == "reset_parameters":
                _send_response("reset_parameters", env.reset_parameters)
            elif cmd.name == "reset":
                all_brain_info = env.reset(cmd.payload[0], cmd.payload[1])
                _send_response("reset", all_brain_info)
            elif cmd.name == "global_done":
                _send_response("global_done", env.global_done)
            elif cmd.name == "close":
                break
    except KeyboardInterrupt:
        print("UnityEnvironment worker: keyboard interrupt")
    finally:
        env.close()


class SubprocessEnvManager(EnvManager):
    def __init__(
        self, env_factory: Callable[[int], BaseUnityEnvironment], n_env: int = 1
    ):
        self.envs: List[UnityEnvWorker] = []
        self.env_last_steps: Dict[int, EnvironmentResponse] = {}
        self.waiting = False
        for worker_id in range(n_env):
            self.envs.append(self.create_worker(worker_id, env_factory))

    def get_last_steps(self):
        return [self.env_last_steps[i] for i in range(len(self.envs))]

    @staticmethod
    def create_worker(
        worker_id: int, env_factory: Callable[[int], BaseUnityEnvironment]
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(
            target=worker, args=(child_conn, pickled_env_factory, worker_id)
        )
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def step(self, steps) -> List[AllBrainInfo]:
        for env_id, step in enumerate(steps):
            env_actions, env_memory, env_text_action, env_value = step
            self.envs[env_id].send(
                "step", (env_actions, env_memory, env_text_action, env_value)
            )

        steps = [self.envs[i].recv().payload for i in range(len(self.envs))]
        for i in range(len(steps)):
            self.env_last_steps[i] = steps[i]
        return steps

    def reset(self, config=None, train_mode=True) -> List[AllBrainInfo]:
        self._broadcast_message("reset", (config, train_mode))
        reset_results = [self.envs[i].recv().payload for i in range(len(self.envs))]
        for i in range(len(reset_results)):
            self.env_last_steps[i] = reset_results[i]
        return reset_results

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        self.envs[0].send("external_brains")
        return self.envs[0].recv().payload

    @property
    def reset_parameters(self):
        self.envs[0].send("reset_parameters")
        return self.envs[0].recv().payload

    def close(self):
        for env in self.envs:
            env.close()

    def _broadcast_message(self, name: str, payload=None):
        for env in self.envs:
            env.send(name, payload)

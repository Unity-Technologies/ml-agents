from typing import *
import cloudpickle

from mlagents.envs import UnityEnvironment
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager, StepInfo
from mlagents.envs import AllBrainInfo, BrainParameters, ActionInfo


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
                all_action_info = cmd.payload
                if env.global_done:
                    all_brain_info = env.reset()
                else:
                    actions = {}
                    memories = {}
                    texts = {}
                    values = {}
                    outputs = {}
                    for brain_name, action_info in all_action_info.items():
                        actions[brain_name] = action_info.action
                        memories[brain_name] = action_info.memory
                        texts[brain_name] = action_info.text
                        values[brain_name] = action_info.value
                        outputs[brain_name] = action_info.outputs
                    all_brain_info = env.step(actions, memories, texts, values)
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
        super().__init__()
        self.envs: List[UnityEnvWorker] = []
        for worker_id in range(n_env):
            self.envs.append(self.create_worker(worker_id, env_factory))
        self.env_last_steps: List[StepInfo] = [None for _ in self.envs]
        self.env_last_action_infos: List[ActionInfo] = [None for _ in self.envs]

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

    def step(self) -> List[StepInfo]:
        env_action_infos: Dict[int, Dict[str, ActionInfo]] = {}
        for env_id in range(len(self.envs)):
            last_step = self.env_last_steps[env_id]
            all_action_info = self._take_step(last_step)
            env_action_infos[env_id] = all_action_info
            self.envs[env_id].send("step", all_action_info)

        step_brain_infos: List[AllBrainInfo] = [
            self.envs[i].recv().payload for i in range(len(self.envs))
        ]
        steps = []
        for i in range(len(step_brain_infos)):
            step_info = StepInfo(
                self.env_last_steps[i].current_all_brain_info,
                step_brain_infos[i],
                env_action_infos[i],
            )
            self.env_last_steps[i] = step_info
            steps.append(step_info)
        return steps

    def reset(self, config=None, train_mode=True) -> List[StepInfo]:
        self._broadcast_message("reset", (config, train_mode))
        reset_results = [self.envs[i].recv().payload for i in range(len(self.envs))]
        for i in range(len(reset_results)):
            self.env_last_steps[i] = StepInfo(None, reset_results[i], None)
        return self.env_last_steps

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

    def _take_step(self, last_step: StepInfo) -> Dict[str, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, brain_info in last_step.current_all_brain_info.items():
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                brain_info
            )
        return all_action_info

from typing import *
import cloudpickle

from mlagents.envs import UnityEnvironment
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager, StepInfo
from mlagents.envs.timers import (
    timed,
    hierarchical_timer,
    reset_timers,
    _global_timer_stack,
)
from mlagents.envs import AllBrainInfo, BrainParameters, ActionInfo


class EnvironmentCommand(NamedTuple):
    name: str
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    name: str
    worker_id: int
    payload: Any


class UnityEnvWorker:
    def __init__(self, process: Process, worker_id: int, conn: Connection):
        self.process = process
        self.worker_id = worker_id
        self.conn = conn
        self.previous_step: StepInfo = StepInfo(None, {}, None)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}

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
                    for brain_name, action_info in all_action_info.items():
                        actions[brain_name] = action_info.action
                        memories[brain_name] = action_info.memory
                        texts[brain_name] = action_info.text
                        values[brain_name] = action_info.value
                    all_brain_info = env.step(actions, memories, texts, values)
                _send_response("step", all_brain_info)
            elif cmd.name == "external_brains":
                _send_response("external_brains", env.external_brains)
            elif cmd.name == "reset_parameters":
                _send_response("reset_parameters", env.reset_parameters)
            elif cmd.name == "reset":
                all_brain_info = env.reset(
                    cmd.payload[0], cmd.payload[1], cmd.payload[2]
                )
                _send_response("reset", all_brain_info)
            elif cmd.name == "global_done":
                _send_response("global_done", env.global_done)
            elif cmd.name == "timers":
                # The timers in this process are independent from all the others.
                # So send back the root timer, then clear them.
                _send_response("timers", _global_timer_stack.root)
                reset_timers()
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
        self.env_workers: List[UnityEnvWorker] = []
        for worker_idx in range(n_env):
            self.env_workers.append(self.create_worker(worker_idx, env_factory))

    def get_last_steps(self):
        return [ew.previous_step for ew in self.env_workers]

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
        for env_worker in self.env_workers:
            all_action_info = self._take_step(env_worker.previous_step)
            env_worker.previous_all_action_info = all_action_info
            env_worker.send("step", all_action_info)

        with hierarchical_timer("step_recv"):
            step_brain_infos: List[AllBrainInfo] = [
                self.env_workers[i].recv().payload for i in range(len(self.env_workers))
            ]
        steps = []
        for i in range(len(step_brain_infos)):
            env_worker = self.env_workers[i]
            step_info = StepInfo(
                env_worker.previous_step.current_all_brain_info,
                step_brain_infos[i],
                env_worker.previous_all_action_info,
            )
            env_worker.previous_step = step_info
            steps.append(step_info)

        # Get timers from the workers, and add them to the "main" timers in this process
        with hierarchical_timer("workers") as main_timer_node:
            for env_worker in self.env_workers:
                env_worker.send("timers")
                worker_timer_node = env_worker.recv().payload
                # TODO store these separately to indicate they ran in parallel?
                main_timer_node.merge(worker_timer_node)

        return steps

    def reset(
        self, config=None, train_mode=True, custom_reset_parameters=None
    ) -> List[StepInfo]:
        self._broadcast_message("reset", (config, train_mode, custom_reset_parameters))
        reset_results = [
            self.env_workers[i].recv().payload for i in range(len(self.env_workers))
        ]
        for i in range(len(reset_results)):
            env_worker = self.env_workers[i]
            env_worker.previous_step = StepInfo(None, reset_results[i], None)
        return list(map(lambda ew: ew.previous_step, self.env_workers))

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        self.env_workers[0].send("external_brains")
        return self.env_workers[0].recv().payload

    @property
    def reset_parameters(self) -> Dict[str, float]:
        self.env_workers[0].send("reset_parameters")
        return self.env_workers[0].recv().payload

    def close(self):
        for env_worker in self.env_workers:
            env_worker.close()

    def _broadcast_message(self, name: str, payload=None):
        for env_worker in self.env_workers:
            env_worker.send(name, payload)

    @timed
    def _take_step(self, last_step: StepInfo) -> Dict[str, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, brain_info in last_step.current_all_brain_info.items():
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                brain_info
            )
        return all_action_info

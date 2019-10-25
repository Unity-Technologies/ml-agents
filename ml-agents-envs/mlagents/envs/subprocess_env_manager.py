import logging
from typing import Dict, NamedTuple, List, Any, Optional, Callable, Set
import cloudpickle

from mlagents.envs.environment import UnityEnvironment
from mlagents.envs.exception import UnityCommunicationException
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
from queue import Empty as EmptyQueueException
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager, EnvironmentStep
from mlagents.envs.timers import (
    TimerNode,
    timed,
    hierarchical_timer,
    reset_timers,
    get_timer_root,
)
from mlagents.envs.brain import AllBrainInfo, BrainParameters
from mlagents.envs.action_info import ActionInfo

logger = logging.getLogger("mlagents.envs")


class EnvironmentCommand(NamedTuple):
    name: str
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    name: str
    worker_id: int
    payload: Any


class StepResponse(NamedTuple):
    all_brain_info: AllBrainInfo
    timer_root: Optional[TimerNode]


class UnityEnvWorker:
    def __init__(self, process: Process, worker_id: int, conn: Connection):
        self.process = process
        self.worker_id = worker_id
        self.conn = conn
        self.previous_step: EnvironmentStep = EnvironmentStep(None, {}, None)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}
        self.waiting = False

    def send(self, name: str, payload: Any = None) -> None:
        try:
            cmd = EnvironmentCommand(name, payload)
            self.conn.send(cmd)
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: send failed.")

    def recv(self) -> EnvironmentResponse:
        try:
            response: EnvironmentResponse = self.conn.recv()
            return response
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: recv failed.")

    def close(self):
        try:
            self.conn.send(EnvironmentCommand("close"))
        except (BrokenPipeError, EOFError):
            logger.debug(
                f"UnityEnvWorker {self.worker_id} got exception trying to close."
            )
            pass
        logger.debug(f"UnityEnvWorker {self.worker_id} joining process.")
        self.process.join()


def worker(
    parent_conn: Connection, step_queue: Queue, pickled_env_factory: str, worker_id: int
) -> None:
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
                # The timers in this process are independent from all the processes and the "main" process
                # So after we send back the root timer, we can safely clear them.
                # Note that we could randomly return timers a fraction of the time if we wanted to reduce
                # the data transferred.
                # TODO get gauges from the workers and merge them in the main process too.
                step_response = StepResponse(all_brain_info, get_timer_root())
                step_queue.put(EnvironmentResponse("step", worker_id, step_response))
                reset_timers()
            elif cmd.name == "external_brains":
                _send_response("external_brains", env.external_brains)
            elif cmd.name == "reset_parameters":
                _send_response("reset_parameters", env.reset_parameters)
            elif cmd.name == "reset":
                all_brain_info = env.reset(
                    cmd.payload[0], cmd.payload[1], cmd.payload[2]
                )
                _send_response("reset", all_brain_info)
            elif cmd.name == "close":
                break
    except (KeyboardInterrupt, UnityCommunicationException):
        logger.info(f"UnityEnvironment worker {worker_id}: environment stopping.")
        step_queue.put(EnvironmentResponse("env_close", worker_id, None))
    finally:
        # If this worker has put an item in the step queue that hasn't been processed by the EnvManager, the process
        # will hang until the item is processed. We avoid this behavior by using Queue.cancel_join_thread()
        # See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.cancel_join_thread for
        # more info.
        logger.debug(f"UnityEnvironment worker {worker_id} closing.")
        step_queue.cancel_join_thread()
        step_queue.close()
        env.close()
        logger.debug(f"UnityEnvironment worker {worker_id} done.")


class SubprocessEnvManager(EnvManager):
    def __init__(
        self, env_factory: Callable[[int], BaseUnityEnvironment], n_env: int = 1
    ):
        super().__init__()
        self.env_workers: List[UnityEnvWorker] = []
        self.step_queue: Queue = Queue()
        for worker_idx in range(n_env):
            self.env_workers.append(
                self.create_worker(worker_idx, self.step_queue, env_factory)
            )

    @staticmethod
    def create_worker(
        worker_id: int,
        step_queue: Queue,
        env_factory: Callable[[int], BaseUnityEnvironment],
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(
            target=worker, args=(child_conn, step_queue, pickled_env_factory, worker_id)
        )
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def _queue_steps(self) -> None:
        for env_worker in self.env_workers:
            if not env_worker.waiting:
                env_action_info = self._take_step(env_worker.previous_step)
                env_worker.previous_all_action_info = env_action_info
                env_worker.send("step", env_action_info)
                env_worker.waiting = True

    def step(self) -> List[EnvironmentStep]:
        # Queue steps for any workers which aren't in the "waiting" state.
        self._queue_steps()

        worker_steps: List[EnvironmentResponse] = []
        step_workers: Set[int] = set()
        # Poll the step queue for completed steps from environment workers until we retrieve
        # 1 or more, which we will then return as StepInfos
        while len(worker_steps) < 1:
            try:
                while True:
                    step = self.step_queue.get_nowait()
                    if step.name == "env_close":
                        raise UnityCommunicationException(
                            "At least one of the environments has closed."
                        )
                    self.env_workers[step.worker_id].waiting = False
                    if step.worker_id not in step_workers:
                        worker_steps.append(step)
                        step_workers.add(step.worker_id)
            except EmptyQueueException:
                pass

        step_infos = self._postprocess_steps(worker_steps)
        return step_infos

    def reset(
        self,
        config: Optional[Dict] = None,
        train_mode: bool = True,
        custom_reset_parameters: Any = None,
    ) -> List[EnvironmentStep]:
        while any(ew.waiting for ew in self.env_workers):
            if not self.step_queue.empty():
                step = self.step_queue.get_nowait()
                self.env_workers[step.worker_id].waiting = False
        # First enqueue reset commands for all workers so that they reset in parallel
        for ew in self.env_workers:
            ew.send("reset", (config, train_mode, custom_reset_parameters))
        # Next (synchronously) collect the reset observations from each worker in sequence
        for ew in self.env_workers:
            ew.previous_step = EnvironmentStep(None, ew.recv().payload, None)
        return list(map(lambda ew: ew.previous_step, self.env_workers))

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        self.env_workers[0].send("external_brains")
        return self.env_workers[0].recv().payload

    @property
    def reset_parameters(self) -> Dict[str, float]:
        self.env_workers[0].send("reset_parameters")
        return self.env_workers[0].recv().payload

    def close(self) -> None:
        logger.debug(f"SubprocessEnvManager closing.")
        self.step_queue.close()
        self.step_queue.join_thread()
        for env_worker in self.env_workers:
            env_worker.close()

    def _postprocess_steps(
        self, env_steps: List[EnvironmentResponse]
    ) -> List[EnvironmentStep]:
        step_infos = []
        timer_nodes = []
        for step in env_steps:
            payload: StepResponse = step.payload
            env_worker = self.env_workers[step.worker_id]
            new_step = EnvironmentStep(
                env_worker.previous_step.current_all_brain_info,
                payload.all_brain_info,
                env_worker.previous_all_action_info,
            )
            step_infos.append(new_step)
            env_worker.previous_step = new_step

            if payload.timer_root:
                timer_nodes.append(payload.timer_root)

        if timer_nodes:
            with hierarchical_timer("workers") as main_timer_node:
                for worker_timer_node in timer_nodes:
                    main_timer_node.merge(
                        worker_timer_node, root_name="worker_root", is_parallel=True
                    )

        return step_infos

    @timed
    def _take_step(self, last_step: EnvironmentStep) -> Dict[str, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, brain_info in last_step.current_all_brain_info.items():
            if brain_name in self.policies:
                all_action_info[brain_name] = self.policies[brain_name].get_action(
                    brain_info
                )
        return all_action_info

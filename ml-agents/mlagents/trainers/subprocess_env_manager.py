from typing import Dict, NamedTuple, List, Any, Optional, Callable, Set
import cloudpickle
import enum

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnityTimeOutException,
    UnityEnvironmentException,
    UnityCommunicatorStoppedException,
)
from multiprocessing import Process, Pipe, Queue
from multiprocessing.connection import Connection
from queue import Empty as EmptyQueueException
from mlagents_envs.base_env import BaseEnv, BehaviorName, BehaviorSpec
from mlagents_envs import logging_util
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep, AllStepResult
from mlagents_envs.timers import (
    TimerNode,
    timed,
    hierarchical_timer,
    reset_timers,
    get_timer_root,
)
from mlagents.trainers.settings import ParameterRandomizationSettings
from mlagents.trainers.action_info import ActionInfo
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
    EngineConfig,
)
from mlagents_envs.side_channel.stats_side_channel import (
    StatsSideChannel,
    EnvironmentStats,
)
from mlagents_envs.side_channel.side_channel import SideChannel


logger = logging_util.get_logger(__name__)


class EnvironmentCommand(enum.Enum):
    STEP = 1
    BEHAVIOR_SPECS = 2
    ENVIRONMENT_PARAMETERS = 3
    RESET = 4
    CLOSE = 5
    ENV_EXITED = 6


class EnvironmentRequest(NamedTuple):
    cmd: EnvironmentCommand
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    cmd: EnvironmentCommand
    worker_id: int
    payload: Any


class StepResponse(NamedTuple):
    all_step_result: AllStepResult
    timer_root: Optional[TimerNode]
    environment_stats: EnvironmentStats


class UnityEnvWorker:
    def __init__(self, process: Process, worker_id: int, conn: Connection):
        self.process = process
        self.worker_id = worker_id
        self.conn = conn
        self.previous_step: EnvironmentStep = EnvironmentStep.empty(worker_id)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}
        self.waiting = False

    def send(self, cmd: EnvironmentCommand, payload: Any = None) -> None:
        try:
            req = EnvironmentRequest(cmd, payload)
            self.conn.send(req)
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: send failed.")

    def recv(self) -> EnvironmentResponse:
        try:
            response: EnvironmentResponse = self.conn.recv()
            if response.cmd == EnvironmentCommand.ENV_EXITED:
                env_exception: Exception = response.payload
                raise env_exception
            return response
        except (BrokenPipeError, EOFError):
            raise UnityCommunicationException("UnityEnvironment worker: recv failed.")

    def close(self):
        try:
            self.conn.send(EnvironmentRequest(EnvironmentCommand.CLOSE))
        except (BrokenPipeError, EOFError):
            logger.debug(
                f"UnityEnvWorker {self.worker_id} got exception trying to close."
            )
            pass
        logger.debug(f"UnityEnvWorker {self.worker_id} joining process.")
        self.process.join()


def worker(
    parent_conn: Connection,
    step_queue: Queue,
    pickled_env_factory: str,
    worker_id: int,
    engine_configuration: EngineConfig,
    log_level: int = logging_util.INFO,
) -> None:
    env_factory: Callable[
        [int, List[SideChannel]], UnityEnvironment
    ] = cloudpickle.loads(pickled_env_factory)
    env_parameters = EnvironmentParametersChannel()
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration(engine_configuration)
    stats_channel = StatsSideChannel()
    env: BaseEnv = None
    # Set log level. On some platforms, the logger isn't common with the
    # main process, so we need to set it again.
    logging_util.set_log_level(log_level)

    def _send_response(cmd_name: EnvironmentCommand, payload: Any) -> None:
        parent_conn.send(EnvironmentResponse(cmd_name, worker_id, payload))

    def _generate_all_results() -> AllStepResult:
        all_step_result: AllStepResult = {}
        for brain_name in env.behavior_specs:
            all_step_result[brain_name] = env.get_steps(brain_name)
        return all_step_result

    try:
        env = env_factory(
            worker_id, [env_parameters, engine_configuration_channel, stats_channel]
        )
        while True:
            req: EnvironmentRequest = parent_conn.recv()
            if req.cmd == EnvironmentCommand.STEP:
                all_action_info = req.payload
                for brain_name, action_info in all_action_info.items():
                    if len(action_info.action) != 0:
                        env.set_actions(brain_name, action_info.action)
                env.step()
                all_step_result = _generate_all_results()
                # The timers in this process are independent from all the processes and the "main" process
                # So after we send back the root timer, we can safely clear them.
                # Note that we could randomly return timers a fraction of the time if we wanted to reduce
                # the data transferred.
                # TODO get gauges from the workers and merge them in the main process too.
                env_stats = stats_channel.get_and_reset_stats()
                step_response = StepResponse(
                    all_step_result, get_timer_root(), env_stats
                )
                step_queue.put(
                    EnvironmentResponse(
                        EnvironmentCommand.STEP, worker_id, step_response
                    )
                )
                reset_timers()
            elif req.cmd == EnvironmentCommand.BEHAVIOR_SPECS:
                _send_response(EnvironmentCommand.BEHAVIOR_SPECS, env.behavior_specs)
            elif req.cmd == EnvironmentCommand.ENVIRONMENT_PARAMETERS:
                for k, v in req.payload.items():
                    if isinstance(v, ParameterRandomizationSettings):
                        v.apply(k, env_parameters)
            elif req.cmd == EnvironmentCommand.RESET:
                env.reset()
                all_step_result = _generate_all_results()
                _send_response(EnvironmentCommand.RESET, all_step_result)
            elif req.cmd == EnvironmentCommand.CLOSE:
                break
    except (
        KeyboardInterrupt,
        UnityCommunicationException,
        UnityTimeOutException,
        UnityEnvironmentException,
        UnityCommunicatorStoppedException,
    ) as ex:
        logger.info(f"UnityEnvironment worker {worker_id}: environment stopping.")
        step_queue.put(
            EnvironmentResponse(EnvironmentCommand.ENV_EXITED, worker_id, ex)
        )
        _send_response(EnvironmentCommand.ENV_EXITED, ex)
    finally:
        # If this worker has put an item in the step queue that hasn't been processed by the EnvManager, the process
        # will hang until the item is processed. We avoid this behavior by using Queue.cancel_join_thread()
        # See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.cancel_join_thread for
        # more info.
        logger.debug(f"UnityEnvironment worker {worker_id} closing.")
        step_queue.cancel_join_thread()
        step_queue.close()
        if env is not None:
            env.close()
        logger.debug(f"UnityEnvironment worker {worker_id} done.")


class SubprocessEnvManager(EnvManager):
    def __init__(
        self,
        env_factory: Callable[[int, List[SideChannel]], BaseEnv],
        engine_configuration: EngineConfig,
        n_env: int = 1,
    ):
        super().__init__()
        self.env_workers: List[UnityEnvWorker] = []
        self.step_queue: Queue = Queue()
        for worker_idx in range(n_env):
            self.env_workers.append(
                self.create_worker(
                    worker_idx, self.step_queue, env_factory, engine_configuration
                )
            )

    @staticmethod
    def create_worker(
        worker_id: int,
        step_queue: Queue,
        env_factory: Callable[[int, List[SideChannel]], BaseEnv],
        engine_configuration: EngineConfig,
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(
            target=worker,
            args=(
                child_conn,
                step_queue,
                pickled_env_factory,
                worker_id,
                engine_configuration,
                logger.level,
            ),
        )
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def _queue_steps(self) -> None:
        for env_worker in self.env_workers:
            if not env_worker.waiting:
                env_action_info = self._take_step(env_worker.previous_step)
                env_worker.previous_all_action_info = env_action_info
                env_worker.send(EnvironmentCommand.STEP, env_action_info)
                env_worker.waiting = True

    def _step(self) -> List[EnvironmentStep]:
        # Queue steps for any workers which aren't in the "waiting" state.
        self._queue_steps()

        worker_steps: List[EnvironmentResponse] = []
        step_workers: Set[int] = set()
        # Poll the step queue for completed steps from environment workers until we retrieve
        # 1 or more, which we will then return as StepInfos
        while len(worker_steps) < 1:
            try:
                while True:
                    step: EnvironmentResponse = self.step_queue.get_nowait()
                    if step.cmd == EnvironmentCommand.ENV_EXITED:
                        env_exception: Exception = step.payload
                        raise env_exception
                    self.env_workers[step.worker_id].waiting = False
                    if step.worker_id not in step_workers:
                        worker_steps.append(step)
                        step_workers.add(step.worker_id)
            except EmptyQueueException:
                pass

        step_infos = self._postprocess_steps(worker_steps)
        return step_infos

    def _reset_env(self, config: Optional[Dict] = None) -> List[EnvironmentStep]:
        while any(ew.waiting for ew in self.env_workers):
            if not self.step_queue.empty():
                step = self.step_queue.get_nowait()
                self.env_workers[step.worker_id].waiting = False
        # Send config to environment
        self.set_env_parameters(config)
        # First enqueue reset commands for all workers so that they reset in parallel
        for ew in self.env_workers:
            ew.send(EnvironmentCommand.RESET, config)
        # Next (synchronously) collect the reset observations from each worker in sequence
        for ew in self.env_workers:
            ew.previous_step = EnvironmentStep(ew.recv().payload, ew.worker_id, {}, {})
        return list(map(lambda ew: ew.previous_step, self.env_workers))

    def set_env_parameters(self, config: Dict = None) -> None:
        """
        Sends environment parameter settings to C# via the
        EnvironmentParametersSidehannel for each worker.
        :param config: Dict of environment parameter keys and values
        """
        for ew in self.env_workers:
            ew.send(EnvironmentCommand.ENVIRONMENT_PARAMETERS, config)

    @property
    def training_behaviors(self) -> Dict[BehaviorName, BehaviorSpec]:
        self.env_workers[0].send(EnvironmentCommand.BEHAVIOR_SPECS)
        return self.env_workers[0].recv().payload

    def close(self) -> None:
        logger.debug("SubprocessEnvManager closing.")
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
                payload.all_step_result,
                step.worker_id,
                env_worker.previous_all_action_info,
                payload.environment_stats,
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
    def _take_step(self, last_step: EnvironmentStep) -> Dict[BehaviorName, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, step_tuple in last_step.current_all_step_result.items():
            if brain_name in self.policies:
                all_action_info[brain_name] = self.policies[brain_name].get_action(
                    step_tuple[0], last_step.worker_id
                )
        return all_action_info

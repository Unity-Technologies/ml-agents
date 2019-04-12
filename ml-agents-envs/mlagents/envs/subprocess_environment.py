from typing import *
import copy
import numpy as np
import cloudpickle

from mlagents.envs import UnityEnvironment
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs import AllBrainInfo, UnityEnvironmentException


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
            self.conn.send(EnvironmentCommand('close'))
        except (BrokenPipeError, EOFError):
            pass
        self.process.join()


def worker(parent_conn: Connection, pickled_env_factory: str, worker_id: int):
    env_factory: Callable[[int], UnityEnvironment] = cloudpickle.loads(pickled_env_factory)
    env = env_factory(worker_id)

    def _send_response(cmd_name, payload):
        parent_conn.send(
            EnvironmentResponse(cmd_name, worker_id, payload)
        )
    try:
        while True:
            cmd: EnvironmentCommand = parent_conn.recv()
            if cmd.name == 'step':
                vector_action, memory, text_action, value = cmd.payload
                all_brain_info = env.step(vector_action, memory, text_action, value)
                _send_response('step', all_brain_info)
            elif cmd.name == 'external_brains':
                _send_response('external_brains', env.external_brains)
            elif cmd.name == 'reset_parameters':
                _send_response('reset_parameters', env.reset_parameters)
            elif cmd.name == 'reset':
                all_brain_info = env.reset(cmd.payload[0], cmd.payload[1])
                _send_response('reset', all_brain_info)
            elif cmd.name == 'global_done':
                _send_response('global_done', env.global_done)
            elif cmd.name == 'close':
                break
    except KeyboardInterrupt:
        print('UnityEnvironment worker: keyboard interrupt')
    finally:
        env.close()


class SubprocessUnityEnvironment(BaseUnityEnvironment):
    def __init__(self,
                 env_factory: Callable[[int], BaseUnityEnvironment],
                 n_env: int = 1):
        self.envs = []
        self.env_agent_counts = {}
        self.waiting = False
        for worker_id in range(n_env):
            self.envs.append(self.create_worker(worker_id, env_factory))

    @staticmethod
    def create_worker(
            worker_id: int,
            env_factory: Callable[[int], BaseUnityEnvironment]
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(target=worker, args=(child_conn, pickled_env_factory, worker_id))
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def step_async(self, vector_action, memory=None, text_action=None, value=None) -> None:
        if self.waiting:
            raise UnityEnvironmentException(
                'Tried to take an environment step bore previous step has completed.'
            )

        agent_counts_cum = {}
        for brain_name in self.env_agent_counts.keys():
            agent_counts_cum[brain_name] = np.cumsum(self.env_agent_counts[brain_name])

        # Split the actions provided by the previous set of agent counts, and send the step
        # commands to the workers.
        for worker_id, env in enumerate(self.envs):
            env_actions = {}
            env_memory = {}
            env_text_action = {}
            env_value = {}
            for brain_name in self.env_agent_counts.keys():
                start_ind = 0
                if worker_id > 0:
                    start_ind = agent_counts_cum[brain_name][worker_id - 1]
                end_ind = agent_counts_cum[brain_name][worker_id]
                if vector_action.get(brain_name) is not None:
                    env_actions[brain_name] = vector_action[brain_name][start_ind:end_ind]
                if memory and memory.get(brain_name) is not None:
                    env_memory[brain_name] = memory[brain_name][start_ind:end_ind]
                if text_action and text_action.get(brain_name) is not None:
                    env_text_action[brain_name] = text_action[brain_name][start_ind:end_ind]
                if value and value.get(brain_name) is not None:
                    env_value[brain_name] = value[brain_name][start_ind:end_ind]

            env.send('step', (env_actions, env_memory, env_text_action, env_value))
        self.waiting = True

    def step_await(self) -> AllBrainInfo:
        if not self.waiting:
            raise UnityEnvironmentException('Tried to await an environment step, but no async step was taken.')

        steps = [self.envs[i].recv() for i in range(len(self.envs))]
        self._get_agent_counts(map(lambda s: s.payload, steps))
        combined_brain_info = self._merge_step_info(steps)
        self.waiting = False
        return combined_brain_info

    def step(self, vector_action=None, memory=None, text_action=None, value=None) -> AllBrainInfo:
        self.step_async(vector_action, memory, text_action, value)
        return self.step_await()

    def reset(self, config=None, train_mode=True) -> AllBrainInfo:
        self._broadcast_message('reset', (config, train_mode))
        reset_results = [self.envs[i].recv() for i in range(len(self.envs))]
        self._get_agent_counts(map(lambda r: r.payload, reset_results))

        return self._merge_step_info(reset_results)

    @property
    def global_done(self):
        self._broadcast_message('global_done')
        dones: List[EnvironmentResponse] = [
            self.envs[i].recv().payload for i in range(len(self.envs))
        ]
        return all(dones)

    @property
    def external_brains(self):
        self.envs[0].send('external_brains')
        return self.envs[0].recv().payload

    @property
    def reset_parameters(self):
        self.envs[0].send('reset_parameters')
        return self.envs[0].recv().payload

    def close(self):
        for env in self.envs:
            env.close()

    def _get_agent_counts(self, step_list: Iterable[AllBrainInfo]):
        for i, step in enumerate(step_list):
            for brain_name, brain_info in step.items():
                if brain_name not in self.env_agent_counts.keys():
                    self.env_agent_counts[brain_name] = [0] * len(self.envs)
                self.env_agent_counts[brain_name][i] = len(brain_info.agents)

    @staticmethod
    def _merge_step_info(env_steps: List[EnvironmentResponse]) -> AllBrainInfo:
        accumulated_brain_info: AllBrainInfo = None
        for env_step in env_steps:
            all_brain_info: AllBrainInfo = env_step.payload
            for brain_name, brain_info in all_brain_info.items():
                for i in range(len(brain_info.agents)):
                    brain_info.agents[i] = str(env_step.worker_id) + '-' + str(brain_info.agents[i])
                if accumulated_brain_info:
                    accumulated_brain_info[brain_name].merge(brain_info)
            if not accumulated_brain_info:
                accumulated_brain_info = copy.deepcopy(all_brain_info)
        return accumulated_brain_info

    def _broadcast_message(self, name: str, payload = None):
        for env in self.envs:
            env.send(name, payload)
import logging

from multiprocessing import Process, Queue
import os
import glob
import shutil
import numpy as np
import yaml
from docopt import docopt
from typing import Any, Callable, Dict, Optional, List


from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.exception import TrainerError
from mlagents.trainers import MetaCurriculumError, MetaCurriculum
from mlagents.envs import UnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.subprocess_env_manager import SubprocessEnvManager
from mlagents.envs.env_manager import EnvManager, StepInfo

from mlagents.envs import UnityEnvironment
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.env_manager import EnvManager, StepInfo
from mlagents.envs.timers import timed, hierarchical_timer
from mlagents.envs import AllBrainInfo, BrainParameters, ActionInfo, BrainInfo

from mlagents.envs.communicator_objects import AgentInfoProto


from abc import ABC, abstractmethod
from typing import Dict

from mlagents.envs import AllBrainInfo, BrainParameters

import tempfile

BRAIN_NAME = "Abby Normal"
OBS_SIZE = 10

class MyDumbEnvironment(BaseUnityEnvironment):

    def __init__(self):
        self._brains: Dict[str, BrainParameters] = {}
        self._brains[BRAIN_NAME] = BrainParameters(
            brain_name=BRAIN_NAME,
            vector_observation_space_size=10,
            num_stacked_vector_observations=1,
            camera_resolutions=[],
            vector_action_space_size=[1],
            vector_action_descriptions=[],
            vector_action_space_type=0,  # "discrete"
        )

    def step(
        self, vector_action=None, memory=None, text_action=None, value=None
    ) -> AllBrainInfo:
        #print("step")
        agent_info = AgentInfoProto(
            stacked_vector_observation=[0.0] * OBS_SIZE
        )
        return {
            BRAIN_NAME: BrainInfo.from_agent_proto(0, [agent_info], self._brains[BRAIN_NAME])
        }

    def reset(self, config=None, train_mode=True) -> AllBrainInfo:
        #print("reset")
        #print("step")
        agent_info = AgentInfoProto(
            stacked_vector_observation=[0.0] * OBS_SIZE
        )
        return {
            BRAIN_NAME: BrainInfo.from_agent_proto(0, [agent_info], self._brains[BRAIN_NAME])
        }

    @property
    def global_done(self):

        return False

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        print("external_brains")

        return self._brains

    @property
    def reset_parameters(self) -> Dict[str, str]:
        print("reset_parameters")
        return {}

    def close(self):
        print("close")
        pass


# class LocalEnvManager(EnvManager):
#     def __init__(self, env: BaseUnityEnvironment):
#         super().__init__()
#         self.env: BaseUnityEnvironment = env
#
#     def step(self) -> List[StepInfo]:
#         all_action_info: Dict[str, ActionInfo] = {}
#         env_step = self.env.step(all_action_info)
#         return []
#
#     def reset(self, config=None, train_mode=True) -> List[StepInfo]:
#         return []
#
#     @property
#     def external_brains(self) -> Dict[str, BrainParameters]:
#         return {}
#
#     @property
#     def reset_parameters(self) -> Dict[str, float]:
#         return {}
#
#     def close(self):
#         pass


class EnvContext:
    def __init__(self, env: BaseUnityEnvironment):
        self.env = env
        self.previous_step: StepInfo = StepInfo(None, {}, None)
        self.previous_all_action_info: Dict[str, ActionInfo] = {}

#Copied from SubprocessEnvManager and removed the subprocess part
class LocalEnvManager(EnvManager):
    def __init__(
        self, envs: List[BaseUnityEnvironment]
    ):
        super().__init__()
        self.env_contexts: List[EnvContext] = [EnvContext(env) for env in envs]

    def get_last_steps(self):
        return [ew.previous_step for ew in self.env_contexts]

    def step(self) -> List[StepInfo]:
        step_brain_infos: List[AllBrainInfo] = []
        for env_worker in self.env_contexts:
            all_action_info = self._take_step(env_worker.previous_step)
            env_worker.previous_all_action_info = all_action_info
            # env_worker.send("step", all_action_info)
            env_worker.env.step() # TODO

            if env_worker.env.global_done:
                all_brain_info = env_worker.env.reset()
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
                all_brain_info = env_worker.env.step(actions, memories, texts, values)
            step_brain_infos.append(all_brain_info)

        steps = []
        for i in range(len(step_brain_infos)):
            env_worker = self.env_contexts[i]
            step_info = StepInfo(
                env_worker.previous_step.current_all_brain_info,
                step_brain_infos[i],
                env_worker.previous_all_action_info,
            )
            env_worker.previous_step = step_info
            steps.append(step_info)
        return steps

    def reset(
        self, config=None, train_mode=True, custom_reset_parameters=None
    ) -> List[StepInfo]:
        reset_results = []
        for worker in self.env_contexts:
            all_brain_info = worker.env.reset(
                config=custom_reset_parameters, train_mode=train_mode
            )
            reset_results.append(all_brain_info)
        for i in range(len(reset_results)):
            env_worker = self.env_contexts[i]
            env_worker.previous_step = StepInfo(None, reset_results[i], None)
        return list(map(lambda ew: ew.previous_step, self.env_contexts))

    @property
    def external_brains(self) -> Dict[str, BrainParameters]:
        return self.env_contexts[0].env.external_brains

    @property
    def reset_parameters(self) -> Dict[str, float]:
        return self.env_contexts[0].env.reset_parameters

    def close(self):
        for env_ctx in self.env_contexts:
            env_ctx.env.close()

    @timed
    def _take_step(self, last_step: StepInfo) -> Dict[str, ActionInfo]:
        all_action_info: Dict[str, ActionInfo] = {}
        for brain_name, brain_info in last_step.current_all_brain_info.items():
            all_action_info[brain_name] = self.policies[brain_name].get_action(
                brain_info
            )
        return all_action_info


config = """
default:
    trainer: ppo
    batch_size: 1024
    beta: 5.0e-3
    buffer_size: 10240
    epsilon: 0.2
    hidden_units: 128
    lambd: 0.95
    learning_rate: 3.0e-4
    max_steps: 5.0e4
    memory_size: 256
    normalize: false
    num_epoch: 3
    num_layers: 2
    time_horizon: 64
    sequence_length: 64
    summary_freq: 1000
    use_recurrent: false
    reward_signals: 
        extrinsic:
            strength: 1.0
            gamma: 0.99

"""

def test_discrete():
    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as dir:
        run_id = "id"
        save_freq = 100
        tc = TrainerController(
            dir,
            dir,
            run_id,
            save_freq,
            meta_curriculum=None,
            load=False,
            train=True,
            keep_checkpoints=1,
            lesson=None,
            training_seed=1337,
            fast_simulation=True,
        )

        # Begin training
        env = MyDumbEnvironment()
        env_manager = LocalEnvManager([env])
        trainer_config  = yaml.safe_load(config)
        tc.start_learning(env_manager, trainer_config)

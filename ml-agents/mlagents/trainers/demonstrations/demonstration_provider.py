import abc
from typing import List

from mlagents_envs.base_env import BehaviorSpec

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import Trajectory


class DemonstrationProvider(abc.ABC):
    @abc.abstractmethod
    def get_behavior_spec(self) -> BehaviorSpec:
        pass

    @abc.abstractmethod
    def get_trajectories(self) -> List[Trajectory]:
        pass

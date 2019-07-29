from abc import ABC, abstractmethod
from typing import List

from mlagents.envs.brain import AgentInfo
from mlagents.envs import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(self, brain_info: List[AgentInfo]) -> ActionInfo:
        pass

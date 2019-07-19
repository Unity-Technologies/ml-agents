from abc import ABC, abstractmethod

from mlagents.envs.brain import BrainInfo
from mlagents.envs.action_info import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        pass

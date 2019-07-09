from abc import ABC, abstractmethod

from mlagents.envs import BrainInfo
from mlagents.envs import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        pass

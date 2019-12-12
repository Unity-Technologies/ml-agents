from abc import ABC, abstractmethod

from mlagents.trainers.brain import BrainInfo
from mlagents.trainers.action_info import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        pass

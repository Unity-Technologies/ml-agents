from abc import ABC, abstractmethod

from mlagentsenvs.envs.brain import BrainInfo
from mlagentsenvs.envs.action_info import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        pass

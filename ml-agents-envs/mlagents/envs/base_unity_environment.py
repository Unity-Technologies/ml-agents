from abc import ABC, abstractmethod
from typing import Dict, List

from mlagents.envs import BrainParameters
from mlagents.envs.brain import AgentInfo


class BaseUnityEnvironment(ABC):
    @abstractmethod
    def step(
        self, vector_action=None, memory=None, text_action=None, value=None
    ) -> List[AgentInfo]:
        pass

    @abstractmethod
    def reset(
        self, config=None, train_mode=True, custom_reset_parameters=None
    ) -> List[AgentInfo]:
        pass

    @property
    @abstractmethod
    def global_done(self):
        pass

    @property
    @abstractmethod
    def external_brains(self) -> Dict[str, BrainParameters]:
        pass

    @property
    @abstractmethod
    def reset_parameters(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def close(self):
        pass

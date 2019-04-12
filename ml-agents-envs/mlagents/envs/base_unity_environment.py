from abc import ABC, abstractmethod
from typing import Dict

from mlagents.envs import AllBrainInfo, BrainParameters


class BaseUnityEnvironment(ABC):
    @abstractmethod
    def step(
        self, vector_action=None, memory=None, text_action=None, value=None
    ) -> AllBrainInfo:
        pass

    @abstractmethod
    def reset(self, config=None, train_mode=True) -> AllBrainInfo:
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
    def reset_parameters(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def close(self):
        pass

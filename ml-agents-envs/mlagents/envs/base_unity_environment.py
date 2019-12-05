from abc import ABC, abstractmethod
from typing import Dict, Optional

from mlagents.envs.brain import AllBrainInfo, BrainParameters


class BaseUnityEnvironment(ABC):
    @abstractmethod
    def step(
        self, vector_action: Optional[Dict] = None, value: Optional[Dict] = None
    ) -> AllBrainInfo:
        pass

    @abstractmethod
    def reset(self) -> AllBrainInfo:
        pass

    @property
    @abstractmethod
    def external_brains(self) -> Dict[str, BrainParameters]:
        pass

    @abstractmethod
    def close(self):
        pass

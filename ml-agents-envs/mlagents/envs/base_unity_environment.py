from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from mlagents.envs.brain import AllBrainInfo, BrainParameters


class BaseUnityEnvironment(ABC):
    @abstractmethod
    def step(
        self,
        vector_action: Optional[Dict] = None,
        text_action: Optional[Dict] = None,
        value: Optional[Dict] = None,
        custom_action: Dict[str, Any] = None,
    ) -> AllBrainInfo:
        pass

    @abstractmethod
    def reset(
        self,
        config: Optional[Dict] = None,
        train_mode: bool = True,
        custom_reset_parameters: Any = None,
    ) -> AllBrainInfo:
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

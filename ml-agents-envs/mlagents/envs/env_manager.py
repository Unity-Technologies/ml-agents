from abc import ABC, abstractmethod
from typing import List, Dict
from mlagents.envs import AllBrainInfo, BrainParameters


class EnvManager(ABC):
    @abstractmethod
    def step(self, steps) -> List[AllBrainInfo]:
        pass

    @abstractmethod
    def reset(self, config=None, train_mode=True) -> List[AllBrainInfo]:
        pass

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

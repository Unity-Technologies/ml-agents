from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Optional
from mlagents.envs import AllBrainInfo, BrainParameters, Policy, ActionInfo


class StepInfo(NamedTuple):
    last_all_brain_info: Optional[AllBrainInfo]
    current_all_brain_info: AllBrainInfo
    all_action_infos: Optional[Dict[str, ActionInfo]]


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[str, Policy] = {}

    def set_policy(self, brain_name: str, policy: Policy):
        self.policies[brain_name] = policy

    @abstractmethod
    def step(self) -> List[StepInfo]:
        pass

    @abstractmethod
    def reset(self, config=None, train_mode=True) -> List[StepInfo]:
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

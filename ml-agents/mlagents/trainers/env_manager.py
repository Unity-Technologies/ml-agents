from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Iterable
from mlagents.trainers.brain import AllBrainInfo, BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo


class EnvironmentStep(NamedTuple):
    previous_all_brain_info: AllBrainInfo
    current_all_brain_info: AllBrainInfo
    brain_name_to_action_info: Dict[str, ActionInfo]

    @property
    def name_behavior_ids(self) -> Iterable[str]:
        return self.brain_name_to_action_info.keys()


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[str, Policy] = {}

    def set_policy(self, brain_name: str, policy: Policy) -> None:
        self.policies[brain_name] = policy

    @abstractmethod
    def step(self) -> List[EnvironmentStep]:
        pass

    @abstractmethod
    def reset(self, config: Dict = None) -> List[EnvironmentStep]:
        pass

    @property
    @abstractmethod
    def external_brains(self) -> Dict[str, BrainParameters]:
        pass

    @property
    @abstractmethod
    def get_properties(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def close(self):
        pass

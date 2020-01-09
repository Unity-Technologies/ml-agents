from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Iterable
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo

AllStepResult = Dict[str, BatchedStepResult]
AllGroupSpec = Dict[str, AgentGroupSpec]


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    brain_name_to_action_info: Dict[str, ActionInfo]

    @property
    def name_behavior_ids(self) -> Iterable[str]:
        return self.current_all_step_result.keys()


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

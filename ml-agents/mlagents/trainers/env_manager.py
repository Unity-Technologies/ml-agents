from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Iterable
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec, AgentGroup
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo

AllStepResult = Dict[AgentGroup, BatchedStepResult]
AllGroupSpec = Dict[AgentGroup, AgentGroupSpec]


class EnvironmentStep(NamedTuple):
    current_all_step_result: AllStepResult
    worker_id: int
    brain_name_to_action_info: Dict[AgentGroup, ActionInfo]

    @property
    def name_behavior_ids(self) -> Iterable[AgentGroup]:
        return self.current_all_step_result.keys()

    @staticmethod
    def empty(worker_id: int) -> "EnvironmentStep":
        return EnvironmentStep({}, worker_id, {})


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[AgentGroup, Policy] = {}

    def set_policy(self, brain_name: AgentGroup, policy: Policy) -> None:
        self.policies[brain_name] = policy

    @abstractmethod
    def step(self) -> List[EnvironmentStep]:
        pass

    @abstractmethod
    def reset(self, config: Dict = None) -> List[EnvironmentStep]:
        pass

    @property
    @abstractmethod
    def external_brains(self) -> Dict[AgentGroup, BrainParameters]:
        pass

    @property
    @abstractmethod
    def get_properties(self) -> Dict[AgentGroup, float]:
        pass

    @abstractmethod
    def close(self):
        pass

from abc import ABC, abstractmethod
from typing import List, Dict, NamedTuple, Optional
from mlagents.envs import BrainParameters, Policy, ActionInfo
from mlagents.envs.brain import AgentInfo


class AgentStep(NamedTuple):
    previous_agent_info: Optional[AgentInfo]
    current_agent_info: AgentInfo
    action_info: Optional[ActionInfo]


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[str, Policy] = {}

    def set_policy(self, brain_name: str, policy: Policy) -> None:
        self.policies[brain_name] = policy

    @abstractmethod
    def step(self) -> List[AgentStep]:
        pass

    @abstractmethod
    def reset(self, config=None, train_mode=True) -> List[AgentStep]:
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

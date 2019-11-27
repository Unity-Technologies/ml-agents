from abc import ABC, abstractmethod
from typing import Any, List, Dict, NamedTuple, Optional
from mlagents.envs.brain import AllBrainInfo, BrainParameters
from mlagents.envs.policy import Policy
from mlagents.envs.action_info import ActionInfo


class EnvironmentStep(NamedTuple):
    previous_all_brain_info: Optional[AllBrainInfo]
    current_all_brain_info: AllBrainInfo
    brain_name_to_action_info: Optional[Dict[str, ActionInfo]]

    def has_actions_for_brain(self, brain_name: str) -> bool:
        return (
            self.brain_name_to_action_info is not None
            and brain_name in self.brain_name_to_action_info
            and self.brain_name_to_action_info[brain_name].outputs is not None
        )


class EnvManager(ABC):
    def __init__(self):
        self.policies: Dict[str, Policy] = {}

    def set_policy(self, brain_name: str, policy: Policy) -> None:
        self.policies[brain_name] = policy

    @abstractmethod
    def step(self) -> List[EnvironmentStep]:
        pass

    @abstractmethod
    def reset(
        self,
        config: Dict = None,
        train_mode: bool = True,
        custom_reset_parameters: Any = None,
    ) -> List[EnvironmentStep]:
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

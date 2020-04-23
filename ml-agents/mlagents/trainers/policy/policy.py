from abc import ABC, abstractmethod

from mlagents_envs.base_env import DecisionSteps
from mlagents.trainers.action_info import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        pass

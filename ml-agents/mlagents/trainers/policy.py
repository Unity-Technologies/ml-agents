from abc import ABC, abstractmethod

from mlagents_envs.base_env import BatchedStepResult
from mlagents.trainers.action_info import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(self, batched_step_result: BatchedStepResult) -> ActionInfo:
        pass

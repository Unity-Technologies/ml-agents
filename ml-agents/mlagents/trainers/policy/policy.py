from abc import abstractmethod

from mlagents_envs.base_env import DecisionSteps
from mlagents_envs.exception import UnityException

from mlagents.trainers.action_info import ActionInfo


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Policy(object):
    def __init__(self, brain, seed):
        self.brain = brain
        self.seed = seed
        self.model_path = None

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        raise NotImplementedError

    @abstractmethod
    def increment_step(self, n_steps):
        pass

    @abstractmethod
    def save_model(self, step):
        pass

# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Deque, Any
import abc

from collections import deque

from mlagents_envs.exception import UnityException
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy

LOGGER = logging.getLogger("mlagents.trainers")


class UnityTrainerException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Trainer(abc.ABC):
    """This class is the base class for the mlagents_envs.trainers"""

    def __init__(
        self,
        brain_name: str,
        trainer_parameters: dict,
        training: bool,
        run_id: str,
        reward_buff_cap: int = 1,
    ):
        """
        Responsible for collecting experiences and training a neural network model.
        :BrainParameters brain: Brain to be trained.
        :dict trainer_parameters: The parameters for the trainer (dictionary).
        :bool training: Whether the trainer is set for training.
        :str run_id: The identifier of the current run
        :int reward_buff_cap:
        """
        self.brain_name = brain_name
        self.run_id = run_id
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.summary_path = trainer_parameters["summary_path"]
        self.stats_reporter = StatsReporter(self.summary_path)
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy_queues: List[AgentManagerQueue[Policy]] = []
        self.trajectory_queues: List[AgentManagerQueue[Trajectory]] = []

    def _dict_to_str(self, param_dict: Dict[str, Any], num_tabs: int) -> str:
        """
        Takes a parameter dictionary and converts it to a human-readable string.
        Recurses if there are multiple levels of dict. Used to print out hyperaparameters.
        param: param_dict: A Dictionary of key, value parameters.
        return: A string version of this dictionary.
        """
        if not isinstance(param_dict, dict):
            return str(param_dict)
        else:
            append_newline = "\n" if num_tabs > 0 else ""
            return append_newline + "\n".join(
                [
                    "\t"
                    + "  " * num_tabs
                    + "{0}:\t{1}".format(
                        x, self._dict_to_str(param_dict[x], num_tabs + 1)
                    )
                    for x in param_dict
                ]
            )

    def __str__(self) -> str:
        return """Hyperparameters for the {0} of brain {1}: \n{2}""".format(
            self.__class__.__name__,
            self.brain_name,
            self._dict_to_str(self.trainer_parameters, 0),
        )

    @property
    @abc.abstractmethod
    def training_progress(self) -> float:
        """
        Returns a float between 0 and 1 indicating how far along in the training progress the Trainer is.
        If 1, the Trainer wasn't training to begin with, or max_steps
        is reached.
        """
        pass

    @property
    def reward_buffer(self) -> Deque[float]:
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    @abc.abstractmethod
    def save_model(self, name_behavior_id: str) -> None:
        """
        Saves the model
        """
        pass

    @abc.abstractmethod
    def export_model(self, name_behavior_id: str) -> None:
        """
        Exports the model
        """
        pass

    @abc.abstractmethod
    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        pass

    @abc.abstractmethod
    def create_policy(self, brain_parameters: BrainParameters) -> TFPolicy:
        """
        Creates policy
        """
        pass

    @abc.abstractmethod
    def add_policy(self, name_behavior_id: str, policy: TFPolicy) -> None:
        """
        Adds policy to trainer
        """
        pass

    @abc.abstractmethod
    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        """
        Gets policy from trainer
        """
        pass

    @abc.abstractmethod
    def advance(self) -> None:
        """
        Steps the trainer, taking in trajectories and updates if ready
        """
        pass

    def publish_policy_queue(self, policy_queue: AgentManagerQueue[Policy]) -> None:
        """
        Adds a policy queue to the list of queues to publish to when this Trainer
        makes a policy update
        :param queue: Policy queue to publish to.
        """
        self.policy_queues.append(policy_queue)

    def subscribe_trajectory_queue(
        self, trajectory_queue: AgentManagerQueue[Trajectory]
    ) -> None:
        """
        Adds a trajectory queue to the list of queues for the trainer injest Trajectories from.
        :param queue: Trajectory queue to publish to.
        """
        self.trajectory_queues.append(trajectory_queue)

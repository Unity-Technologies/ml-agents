# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Deque, Any
import abc

from collections import deque

from mlagents.model_serialization import export_policy_model, SerializationSettings
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.exception import UnityTrainerException

logger = logging.getLogger("mlagents.trainers")


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
        self.param_keys: List[str] = []
        self.brain_name = brain_name
        self.run_id = run_id
        self.trainer_parameters = trainer_parameters
        self.summary_path = trainer_parameters["summary_path"]
        self._stats_reporter = StatsReporter(self.summary_path)
        self.is_training = training
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy_queues: List[AgentManagerQueue[Policy]] = []
        self.trajectory_queues: List[AgentManagerQueue[Trajectory]] = []
        self.step: int = 0
        self.summary_freq = self.trainer_parameters["summary_freq"]
        self.next_summary_step = self.summary_freq

    @property
    def stats_reporter(self):
        """
        Returns the stats reporter associated with this Trainer.
        """
        return self._stats_reporter

    def _check_param_keys(self):
        for k in self.param_keys:
            if k not in self.trainer_parameters:
                raise UnityTrainerException(
                    "The hyper-parameter {0} could not be found for the {1} trainer of "
                    "brain {2}.".format(k, self.__class__, self.brain_name)
                )

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self) -> int:
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return int(float(self.trainer_parameters["max_steps"]))

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.step

    @property
    def should_still_train(self) -> bool:
        """
        Returns whether or not the trainer should train. A Trainer could
        stop training if it wasn't training to begin with, or if max_steps
        is reached.
        """
        return self.is_training and self.get_step <= self.get_max_steps

    @property
    def reward_buffer(self) -> Deque[float]:
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    def save_model(self, name_behavior_id: str) -> None:
        """
        Saves the model
        """
        self.get_policy(name_behavior_id).save_model(self.get_step)

    def export_model(self, name_behavior_id: str) -> None:
        """
        Exports the model
        """
        policy = self.get_policy(name_behavior_id)
        settings = SerializationSettings(policy.model_path, policy.brain.brain_name)
        export_policy_model(settings, policy.graph, policy.sess)

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
        Adds policy to trainer.
        """
        pass

    @abc.abstractmethod
    def get_policy(self, name_behavior_id: str) -> TFPolicy:
        """
        Gets policy from trainer.
        """
        pass

    @abc.abstractmethod
    def advance(self) -> None:
        """
        Advances the trainer. Typically, this means grabbing trajectories
        from all subscribed trajectory queues (self.trajectory_queues), and updating
        a policy using the steps in them, and if needed pushing a new policy onto the right
        policy queues (self.policy_queues).
        """
        pass

    def publish_policy_queue(self, policy_queue: AgentManagerQueue[Policy]) -> None:
        """
        Adds a policy queue to the list of queues to publish to when this Trainer
        makes a policy update
        :param policy_queue: Policy queue to publish to.
        """
        self.policy_queues.append(policy_queue)

    def subscribe_trajectory_queue(
        self, trajectory_queue: AgentManagerQueue[Trajectory]
    ) -> None:
        """
        Adds a trajectory queue to the list of queues for the trainer to ingest Trajectories from.
        :param trajectory_queue: Trajectory queue to read from.
        """
        self.trajectory_queues.append(trajectory_queue)

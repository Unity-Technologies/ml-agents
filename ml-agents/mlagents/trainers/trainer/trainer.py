# # Unity ML-Agents Toolkit
from typing import List, Deque, Dict
import abc
from collections import deque

from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.policy import Policy
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings


logger = get_logger(__name__)


class Trainer(abc.ABC):
    """This class is the base class for the mlagents_envs.trainers"""

    def __init__(
        self,
        brain_name: str,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        artifact_path: str,
        reward_buff_cap: int = 1,
    ):
        """
        Responsible for collecting experiences and training a neural network model.
        :param brain_name: Brain name of brain to be trained.
        :param trainer_settings: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param artifact_path: The directory within which to store artifacts from this trainer
        :param reward_buff_cap:
        """
        self.brain_name = brain_name
        self.trainer_settings = trainer_settings
        self._threaded = trainer_settings.threaded
        self._stats_reporter = StatsReporter(brain_name)
        self.is_training = training
        self.load = load
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy_queues: List[AgentManagerQueue[Policy]] = []
        self.trajectory_queues: List[AgentManagerQueue[Trajectory]] = []
        self._step: int = 0
        self.artifact_path = artifact_path
        self.summary_freq = self.trainer_settings.summary_freq
        self.policies: Dict[str, Policy] = {}

    @property
    def stats_reporter(self):
        """
        Returns the stats reporter associated with this Trainer.
        """
        return self._stats_reporter

    @property
    def parameters(self) -> TrainerSettings:
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_settings

    @property
    def get_max_steps(self) -> int:
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return self.trainer_settings.max_steps

    @property
    def get_step(self) -> int:
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self._step

    @property
    def threaded(self) -> bool:
        """
        Whether or not to run the trainer in a thread. True allows the trainer to
        update the policy while the environment is taking steps. Set to False to
        enforce strict on-policy updates (i.e. don't update the policy when taking steps.)
        """
        return self._threaded

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

    @abc.abstractmethod
    def save_model(self) -> None:
        """
        Saves model file(s) for the policy or policies associated with this trainer.
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
    def create_policy(
        self,
        parsed_behavior_id: BehaviorIdentifiers,
        behavior_spec: BehaviorSpec,
    ) -> Policy:
        """
        Creates a Policy object
        """
        pass

    @abc.abstractmethod
    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy
    ) -> None:
        """
        Adds policy to trainer.
        """
        pass

    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy associated with name_behavior_id
        :param name_behavior_id: Fully qualified behavior name
        :return: Policy associated with name_behavior_id
        """
        return self.policies[name_behavior_id]

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

    @staticmethod
    def get_trainer_name() -> str:
        raise NotImplementedError

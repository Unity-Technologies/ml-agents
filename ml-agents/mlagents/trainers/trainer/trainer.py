# # Unity ML-Agents Toolkit
from typing import List, Deque
import abc

from collections import deque

from mlagents_envs.logging_util import get_logger
from mlagents.model_serialization import export_policy_model, SerializationSettings
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.brain import BrainParameters
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
        run_id: str,
        reward_buff_cap: int = 1,
    ):
        """
        Responsible for collecting experiences and training a neural network model.
        :BrainParameters brain: Brain to be trained.
        :dict trainer_settings: The parameters for the trainer (dictionary).
        :bool training: Whether the trainer is set for training.
        :str run_id: The identifier of the current run
        :int reward_buff_cap:
        """
        self.param_keys: List[str] = []
        self.brain_name = brain_name
        self.run_id = run_id
        self.trainer_settings = trainer_settings
        self._threaded = trainer_settings.threaded
        self._stats_reporter = StatsReporter(brain_name)
        self.is_training = training
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy_queues: List[AgentManagerQueue[Policy]] = []
        self.trajectory_queues: List[AgentManagerQueue[Trajectory]] = []
        self.step: int = 0
        self.summary_freq = self.trainer_settings.summary_freq
        self.next_summary_step = self.summary_freq

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
        return self.step

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
    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, brain_parameters: BrainParameters
    ) -> TFPolicy:
        """
        Creates policy
        """
        pass

    @abc.abstractmethod
    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: TFPolicy
    ) -> None:
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

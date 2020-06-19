# # Unity ML-Agents Toolkit
from typing import List, Deque, Union, Dict, Any
import os
import abc
from numpy import mean

from enum import Enum
from collections import deque
import attr
from collections import defaultdict

from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import timed
from mlagents.model_serialization import export_policy_model, SerializationSettings
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.policy import Policy
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.training_status import GlobalTrainingStatus


logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class Checkpoint:
    steps: int
    file_path: str
    reward: int


class CheckpointType(Enum):
    CHECKPOINT = "checkpoint"
    STEPS = "steps"
    FINAL_PATH = "final_model_path"
    REWARD = "reward"


class CheckpointManagerClass:
    checkpoints_saved: Dict[str, Dict[str, Any]] = defaultdict(lambda: {})

    @staticmethod
    def save_checkpoints():
        GlobalTrainingStatus.saved_state.update(
            CheckpointManagerClass.checkpoints_saved
        )

    @staticmethod
    def set_parameter_state(category: str, key: CheckpointType, value: Any) -> None:
        CheckpointManagerClass.checkpoints_saved[category][key.value] = value

    @staticmethod
    def append_to_parameter_state(
        category: str, key: CheckpointType, value: Any
    ) -> None:
        CheckpointManagerClass.checkpoints_saved[category][key.value].append(value)

    @staticmethod
    def get_parameter_state(category: str, key: CheckpointType) -> Any:
        return CheckpointManagerClass.checkpoints_saved[category].get(key.value, None)

    @staticmethod
    def remove_checkpoint(checkpoint: Dict[str, Any]) -> None:
        """
        Removes a checkpoint stored in checkpoint_list.
        If checkpoint cannot be found, no action is done.
        :param checkpoint: A checkpoint stored in checkpoint_list
        """
        file_path: str = checkpoint["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed checkpoint model {file_path}.")
        else:
            logger.info(f"Checkpoint at {file_path} could not be found.")
        return

    @staticmethod
    def manage_checkpoint_list(category: str, keep_checkpoints: int) -> None:
        """
        Ensures that the number of checkpoints stored are within the number
        of checkpoints the user defines. If the limit is hit, checkpoints are
        removed to create room for the next checkpoint to be inserted.
        :param category: The category (usually behavior name) of the parameter.
        :param keep_checkpoints: Number of checkpoints to record (user-defined).
        """
        if not CheckpointManagerClass.get_parameter_state(
            category, CheckpointType.CHECKPOINT
        ):
            CheckpointManagerClass.set_parameter_state(
                category, CheckpointType.CHECKPOINT, []
            )
        checkpoint_list = CheckpointManagerClass.get_parameter_state(
            category, CheckpointType.CHECKPOINT
        )
        num_checkpoints = len(checkpoint_list)
        while num_checkpoints >= keep_checkpoints:
            if keep_checkpoints <= 0:
                break
            CheckpointManagerClass.remove_checkpoint(checkpoint_list.pop(0))
            num_checkpoints = len(checkpoint_list)
        return

    @staticmethod
    def track_checkpoint_info(
        category: str, value: Dict[str, Union[int, str]], keep_checkpoints: int
    ) -> None:
        """
        Make room for new checkpoint if needed and insert new checkpoint information.
        :param category: The category (usually behavior name) of the parameter.
        :param value: The new checkpoint to be recorded.
        :param keep_checkpoints: Number of checkpoints to record (user-defined).
        """
        CheckpointManagerClass.manage_checkpoint_list(category, keep_checkpoints)
        CheckpointManagerClass.append_to_parameter_state(
            category, CheckpointType.CHECKPOINT, value
        )
        return

    @staticmethod
    def track_final_model_info(
        category: str, final_model_path: str, keep_checkpoints: int, mean_reward: int
    ) -> None:
        """
        Ensures number of checkpoints stored is within the max number of checkpoints
        defined by the user and finally stores the information about the final
        model (or intermediate model if training is interrupted).
        :param category: The category (usually behavior name) of the parameter.
        :param final_model_path: The file path of the final model.
        :param keep_checkpoints: Number of checkpoints to record (user-defined).
        """
        CheckpointManagerClass.manage_checkpoint_list(category, keep_checkpoints)
        CheckpointManagerClass.set_parameter_state(
            category, CheckpointType.FINAL_PATH, final_model_path
        )
        CheckpointManagerClass.set_parameter_state(
            category, CheckpointType.REWARD, mean_reward
        )
        GlobalTrainingStatus.update_parameter_state(
            CheckpointManagerClass.checkpoints_saved
        )
        return


class Trainer(abc.ABC):
    """This class is the base class for the mlagents_envs.trainers"""

    def __init__(
        self,
        brain_name: str,
        trainer_settings: TrainerSettings,
        training: bool,
        artifact_path: str,
        reward_buff_cap: int = 1,
    ):
        """
        Responsible for collecting experiences and training a neural network model.
        :BrainParameters brain: Brain to be trained.
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
        self._reward_buffer: Deque[float] = deque(maxlen=reward_buff_cap)
        self.policy_queues: List[AgentManagerQueue[Policy]] = []
        self.trajectory_queues: List[AgentManagerQueue[Trajectory]] = []
        self.step: int = 0
        self.artifact_path = artifact_path
        self.summary_freq = self.trainer_settings.summary_freq

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

    @timed
    def save_model(self, name_behavior_id: str) -> None:
        """
        Saves the model
        """
        self.get_policy(name_behavior_id).save_model(self.get_step)

    def export_model(self, name_behavior_id: str, is_checkpoint: bool = False) -> None:
        """
        Exports the model
        """
        policy = self.get_policy(name_behavior_id)
        measure_val = mean(self.reward_buffer)
        if is_checkpoint:
            checkpoint_path = f"{self.brain_name}-{self.get_step}"
            settings = SerializationSettings(
                policy.model_path, self.brain_name, checkpoint_path
            )
            # Store steps and file_path
            new_checkpoint = Checkpoint(
                int(self.get_step),
                os.path.join(settings.model_path, f"{settings.checkpoint_path}.nn"),
                measure_val,
            )
            # Record checkpoint information
            CheckpointManagerClass.track_checkpoint_info(
                name_behavior_id, attr.asdict(new_checkpoint), policy.keep_checkpoints
            )
        else:
            # Extracting brain name for consistent name_behavior_id
            settings = SerializationSettings(policy.model_path, self.brain_name)
            # Record final model information
            CheckpointManagerClass.track_final_model_info(
                self.brain_name,
                f"{settings.model_path}.nn",
                policy.keep_checkpoints,
                measure_val,
            )
        export_policy_model(settings, policy.graph, policy.sess, is_checkpoint)


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

# # Unity ML-Agents Toolkit
from typing import Union, Dict, Any
import os
from enum import Enum
import attr
from collections import defaultdict
from mlagents.trainers.training_status import GlobalTrainingStatus
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class Checkpoint:
    steps: int
    file_path: str
    reward: float


class CheckpointType(Enum):
    CHECKPOINT = "checkpoint"
    STEPS = "steps"
    FINAL_PATH = "final_model_path"
    REWARD = "reward"


class CheckpointManager:
    checkpoints_saved: Dict[str, Dict[str, Any]] = defaultdict(lambda: {})

    @staticmethod
    def save_checkpoints():
        GlobalTrainingStatus.saved_state.update(
            CheckpointManager.checkpoints_saved
        )

    @staticmethod
    def set_parameter_state(category: str, key: CheckpointType, value: Any) -> None:
        CheckpointManager.checkpoints_saved[category][key.value] = value

    @staticmethod
    def append_to_parameter_state(
        category: str, key: CheckpointType, value: Any
    ) -> None:
        CheckpointManager.checkpoints_saved[category][key.value].append(value)

    @staticmethod
    def get_parameter_state(category: str, key: CheckpointType) -> Any:
        return CheckpointManager.checkpoints_saved[category].get(key.value, None)

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
        if not CheckpointManager.get_parameter_state(
            category, CheckpointType.CHECKPOINT
        ):
            CheckpointManager.set_parameter_state(
                category, CheckpointType.CHECKPOINT, []
            )
        checkpoint_list = CheckpointManager.get_parameter_state(
            category, CheckpointType.CHECKPOINT
        )
        num_checkpoints = len(checkpoint_list)
        while num_checkpoints >= keep_checkpoints:
            if keep_checkpoints <= 0:
                break
            CheckpointManager.remove_checkpoint(checkpoint_list.pop(0))
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
        CheckpointManager.manage_checkpoint_list(category, keep_checkpoints)
        CheckpointManager.append_to_parameter_state(
            category, CheckpointType.CHECKPOINT, value
        )
        return

    @staticmethod
    def track_final_model_info(
        category: str, final_model_path: str, keep_checkpoints: int, mean_reward: float
    ) -> None:
        """
        Ensures number of checkpoints stored is within the max number of checkpoints
        defined by the user and finally stores the information about the final
        model (or intermediate model if training is interrupted).
        :param category: The category (usually behavior name) of the parameter.
        :param final_model_path: The file path of the final model.
        :param keep_checkpoints: Number of checkpoints to record (user-defined).
        """
        CheckpointManager.manage_checkpoint_list(category, keep_checkpoints)
        CheckpointManager.set_parameter_state(
            category, CheckpointType.FINAL_PATH, final_model_path
        )
        CheckpointManager.set_parameter_state(
            category, CheckpointType.REWARD, mean_reward
        )
        GlobalTrainingStatus.update_parameter_state(
            CheckpointManager.checkpoints_saved
        )
        return

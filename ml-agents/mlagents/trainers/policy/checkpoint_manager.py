# # Unity ML-Agents Toolkit
from typing import Dict, Any, Optional, List
import os
import attr
from mlagents.trainers.training_status import GlobalTrainingStatus, StatusType
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class Checkpoint:
    steps: int
    file_path: str
    reward: Optional[float]
    creation_time: float


class CheckpointManager:
    @staticmethod
    def get_checkpoints(behavior_name: str) -> List[Dict[str, Any]]:
        checkpoint_list = GlobalTrainingStatus.get_parameter_state(
            behavior_name, StatusType.CHECKPOINTS
        )
        if not checkpoint_list:
            checkpoint_list = []
            GlobalTrainingStatus.set_parameter_state(
                behavior_name, StatusType.CHECKPOINTS, checkpoint_list
            )
        return checkpoint_list

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

    @classmethod
    def manage_checkpoint_list(
        cls, behavior_name: str, keep_checkpoints: int
    ) -> List[Dict[str, Any]]:
        """
        Ensures that the number of checkpoints stored are within the number
        of checkpoints the user defines. If the limit is hit, checkpoints are
        removed to create room for the next checkpoint to be inserted.

        :param behavior_name: The behavior name whose checkpoints we will mange.
        :param keep_checkpoints: Number of checkpoints to record (user-defined).
        """
        checkpoints = cls.get_checkpoints(behavior_name)
        while len(checkpoints) >= keep_checkpoints:
            if (keep_checkpoints <= 0) or (len(checkpoints) == 0):
                break
            CheckpointManager.remove_checkpoint(checkpoints.pop(0))
        return checkpoints

    @classmethod
    def track_checkpoint_info(
        cls, behavior_name: str, new_checkpoint: Checkpoint, keep_checkpoints: int
    ) -> None:
        """
        Make room for new checkpoint if needed and insert new checkpoint information.
        :param behavior_name: Behavior name for the checkpoint.
        :param new_checkpoint: The new checkpoint to be recorded.
        :param keep_checkpoints: Number of checkpoints to record (user-defined).
        """
        checkpoints = cls.manage_checkpoint_list(behavior_name, keep_checkpoints)
        new_checkpoint_dict = attr.asdict(new_checkpoint)
        checkpoints.append(new_checkpoint_dict)

    @classmethod
    def track_final_model_info(
        cls, behavior_name: str, final_model: Checkpoint
    ) -> None:
        """
        Ensures number of checkpoints stored is within the max number of checkpoints
        defined by the user and finally stores the information about the final
        model (or intermediate model if training is interrupted).
        :param behavior_name: Behavior name of the model.
        :param final_model: Checkpoint information for the final model.
        """
        final_model_dict = attr.asdict(final_model)
        GlobalTrainingStatus.set_parameter_state(
            behavior_name, StatusType.FINAL_MODEL, final_model_dict
        )

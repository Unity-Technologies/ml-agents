from typing import Dict, Any
from enum import Enum
from collections import defaultdict
import json
import attr
import cattr

from mlagents.tf_utils import tf
from mlagents_envs.logging_util import get_logger
from mlagents.trainers import __version__
from mlagents.trainers.exception import TrainerError

logger = get_logger(__name__)

STATUS_FORMAT_VERSION = "0.1.0"


class StatusType(Enum):
    LESSON_NUM = "lesson_num"
    STATS_METADATA = "metadata"


@attr.s(auto_attribs=True)
class StatusMetaData:
    stats_format_version: str = STATUS_FORMAT_VERSION
    mlagents_version: str = __version__
    tensorflow_version: str = tf.__version__

    def to_dict(self) -> Dict[str, str]:
        return cattr.unstructure(self)

    @staticmethod
    def from_dict(import_dict: Dict[str, str]) -> "StatusMetaData":
        return cattr.structure(import_dict, StatusMetaData)

    def check_compatibility(self, other: "StatusMetaData") -> None:
        """
        Check compatibility with a loaded StatsMetaData and warn the user
        if versions mismatch. This is used for resuming from old checkpoints.
        """
        # This should cover all stats version mismatches as well.
        if self.mlagents_version != other.mlagents_version:
            logger.warning(
                "Checkpoint was loaded from a different version of ML-Agents. Some things may not resume properly."
            )
        if self.tensorflow_version != other.tensorflow_version:
            logger.warning(
                "Tensorflow checkpoint was saved with a different version of Tensorflow. Model may not resume properly."
            )


class GlobalTrainingStatus:
    """
    GlobalTrainingStatus class that contains static methods to save global training status and
    load it on a resume. These are values that might be needed for the training resume that
    cannot/should not be captured in a model checkpoint, such as curriclum lesson.
    """

    saved_state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {})

    @staticmethod
    def load_state(path: str) -> None:
        """
        Load a JSON file that contains saved state.
        :param path: Path to the JSON file containing the state.
        """
        try:
            with open(path, "r") as f:
                loaded_dict = json.load(f)
            # Compare the metadata
            _metadata = loaded_dict[StatusType.STATS_METADATA.value]
            StatusMetaData.from_dict(_metadata).check_compatibility(StatusMetaData())
            # Update saved state.
            GlobalTrainingStatus.saved_state.update(loaded_dict)
        except FileNotFoundError:
            logger.warning(
                "Training status file not found. Not all functions will resume properly."
            )
        except KeyError:
            raise TrainerError(
                "Metadata not found, resuming from an incompatible version of ML-Agents."
            )

    @staticmethod
    def save_state(path: str) -> None:
        """
        Save a JSON file that contains saved state.
        :param path: Path to the JSON file containing the state.
        """
        GlobalTrainingStatus.saved_state[
            StatusType.STATS_METADATA.value
        ] = StatusMetaData().to_dict()
        with open(path, "w") as f:
            json.dump(GlobalTrainingStatus.saved_state, f, indent=4)

    @staticmethod
    def set_parameter_state(category: str, key: StatusType, value: Any) -> None:
        """
        Stores an arbitrary-named parameter in the global saved state.
        :param category: The category (usually behavior name) of the parameter.
        :param key: The parameter, e.g. lesson number.
        :param value: The value.
        """
        GlobalTrainingStatus.saved_state[category][key.value] = value

    @staticmethod
    def get_parameter_state(category: str, key: StatusType) -> Any:
        """
        Loads an arbitrary-named parameter from training_status.json.
        If not found, returns None.
        :param category: The category (usually behavior name) of the parameter.
        :param key: The statistic, e.g. lesson number.
        :param value: The value.
        """
        return GlobalTrainingStatus.saved_state[category].get(key.value, None)

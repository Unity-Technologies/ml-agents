import copy
import json
import hmac
import hashlib
import sys
from typing import Optional, Dict
import mlagents_envs
import mlagents.trainers
from mlagents import torch_utils
from mlagents.trainers.settings import RewardSignalType
from mlagents_envs.exception import UnityCommunicationException
from mlagents_envs.side_channel import (
    IncomingMessage,
    OutgoingMessage,
    DefaultTrainingAnalyticsSideChannel,
)
from mlagents_envs.communicator_objects.training_analytics_pb2 import (
    TrainingEnvironmentInitialized,
    TrainingBehaviorInitialized,
)
from google.protobuf.any_pb2 import Any

from mlagents.trainers.settings import TrainerSettings, RunOptions


class TrainingAnalyticsSideChannel(DefaultTrainingAnalyticsSideChannel):
    """
    Side channel that sends information about the training to the Unity environment so it can be logged.
    """

    __vendorKey: str = "unity.ml-agents"

    def __init__(self) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/TrainingAnalyticsSideChannel")
        # UUID('b664a4a9-d86f-5a5f-95cb-e8353a7e8356')
        # Use the same uuid as the parent side channel
        super().__init__()
        self.run_options: Optional[RunOptions] = None

    @classmethod
    def _hash(cls, data: str) -> str:
        res = hmac.new(
            cls.__vendorKey.encode("utf-8"), data.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return res

    def on_message_received(self, msg: IncomingMessage) -> None:
        raise UnityCommunicationException(
            "The TrainingAnalyticsSideChannel received a message from Unity, "
            "this should not have happened."
        )

    @classmethod
    def _sanitize_run_options(cls, config: RunOptions) -> Dict[str, Any]:
        res = copy.deepcopy(config.as_dict())

        # Filter potentially PII behavior names
        if "behaviors" in res and res["behaviors"]:
            res["behaviors"] = {cls._hash(k): v for (k, v) in res["behaviors"].items()}
            for (k, v) in res["behaviors"].items():
                if "init_path" in v and v["init_path"] is not None:
                    hashed_path = cls._hash(v["init_path"])
                    res["behaviors"][k]["init_path"] = hashed_path
                if "demo_path" in v and v["demo_path"] is not None:
                    hashed_path = cls._hash(v["demo_path"])
                    res["behaviors"][k]["demo_path"] = hashed_path

        # Filter potentially PII curriculum and behavior names from Checkpoint Settings
        if "environment_parameters" in res and res["environment_parameters"]:
            res["environment_parameters"] = {
                cls._hash(k): v for (k, v) in res["environment_parameters"].items()
            }
            for (curriculumName, curriculum) in res["environment_parameters"].items():
                updated_lessons = []
                for lesson in curriculum["curriculum"]:
                    new_lesson = copy.deepcopy(lesson)
                    if "name" in lesson:
                        new_lesson["name"] = cls._hash(lesson["name"])
                    if (
                        "completion_criteria" in lesson
                        and lesson["completion_criteria"] is not None
                    ):
                        new_lesson["completion_criteria"]["behavior"] = cls._hash(
                            new_lesson["completion_criteria"]["behavior"]
                        )
                    updated_lessons.append(new_lesson)
                res["environment_parameters"][curriculumName][
                    "curriculum"
                ] = updated_lessons

        # Filter potentially PII filenames from Checkpoint Settings
        if "checkpoint_settings" in res and res["checkpoint_settings"] is not None:
            if (
                "initialize_from" in res["checkpoint_settings"]
                and res["checkpoint_settings"]["initialize_from"] is not None
            ):
                res["checkpoint_settings"]["initialize_from"] = cls._hash(
                    res["checkpoint_settings"]["initialize_from"]
                )
            if (
                "results_dir" in res["checkpoint_settings"]
                and res["checkpoint_settings"]["results_dir"] is not None
            ):
                res["checkpoint_settings"]["results_dir"] = hash(
                    res["checkpoint_settings"]["results_dir"]
                )

        return res

    def environment_initialized(self, run_options: RunOptions) -> None:
        self.run_options = run_options
        # Tuple of (major, minor, patch)
        vi = sys.version_info
        env_params = run_options.environment_parameters
        sanitized_run_options = self._sanitize_run_options(run_options)

        msg = TrainingEnvironmentInitialized(
            python_version=f"{vi[0]}.{vi[1]}.{vi[2]}",
            mlagents_version=mlagents.trainers.__version__,
            mlagents_envs_version=mlagents_envs.__version__,
            torch_version=torch_utils.torch.__version__,
            torch_device_type=torch_utils.default_device().type,
            num_envs=run_options.env_settings.num_envs,
            num_environment_parameters=len(env_params) if env_params else 0,
            run_options=json.dumps(sanitized_run_options),
        )

        any_message = Any()
        any_message.Pack(msg)

        env_init_msg = OutgoingMessage()
        env_init_msg.set_raw_bytes(any_message.SerializeToString())
        super().queue_message_to_send(env_init_msg)

    @classmethod
    def _sanitize_trainer_settings(cls, config: TrainerSettings) -> Dict[str, Any]:
        config_dict = copy.deepcopy(config.as_dict())
        if "init_path" in config_dict and config_dict["init_path"] is not None:
            hashed_path = cls._hash(config_dict["init_path"])
            config_dict["init_path"] = hashed_path
        if "demo_path" in config_dict and config_dict["demo_path"] is not None:
            hashed_path = cls._hash(config_dict["demo_path"])
            config_dict["demo_path"] = hashed_path
        return config_dict

    def training_started(self, behavior_name: str, config: TrainerSettings) -> None:
        raw_config = self._sanitize_trainer_settings(config)
        msg = TrainingBehaviorInitialized(
            behavior_name=self._hash(behavior_name),
            trainer_type=config.trainer_type.value,
            extrinsic_reward_enabled=(
                RewardSignalType.EXTRINSIC in config.reward_signals
            ),
            gail_reward_enabled=(RewardSignalType.GAIL in config.reward_signals),
            curiosity_reward_enabled=(
                RewardSignalType.CURIOSITY in config.reward_signals
            ),
            rnd_reward_enabled=(RewardSignalType.RND in config.reward_signals),
            behavioral_cloning_enabled=config.behavioral_cloning is not None,
            recurrent_enabled=config.network_settings.memory is not None,
            visual_encoder=config.network_settings.vis_encode_type.value,
            num_network_layers=config.network_settings.num_layers,
            num_network_hidden_units=config.network_settings.hidden_units,
            trainer_threaded=config.threaded,
            self_play_enabled=config.self_play is not None,
            curriculum_enabled=self._behavior_uses_curriculum(behavior_name),
            config=json.dumps(raw_config),
        )

        any_message = Any()
        any_message.Pack(msg)

        training_start_msg = OutgoingMessage()
        training_start_msg.set_raw_bytes(any_message.SerializeToString())

        super().queue_message_to_send(training_start_msg)

    def _behavior_uses_curriculum(self, behavior_name: str) -> bool:
        if not self.run_options or not self.run_options.environment_parameters:
            return False

        for param_settings in self.run_options.environment_parameters.values():
            for lesson in param_settings.curriculum:
                cc = lesson.completion_criteria
                if cc and cc.behavior == behavior_name:
                    return True

        return False

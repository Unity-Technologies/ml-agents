import sys
from typing import Optional
import uuid
import mlagents_envs
import mlagents.trainers
from mlagents import torch_utils
from mlagents.trainers.settings import RewardSignalType
from mlagents_envs.exception import UnityCommunicationException
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.communicator_objects.training_analytics_pb2 import (
    TrainingEnvironmentInitialized,
    TrainingBehaviorInitialized,
)
from google.protobuf.any_pb2 import Any

from mlagents.trainers.settings import TrainerSettings, RunOptions


class TrainingAnalyticsSideChannel(SideChannel):
    """
    Side channel that sends information about the training to the Unity environment so it can be logged.
    """

    def __init__(self) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/TrainingAnalyticsSideChannel")
        # UUID('b664a4a9-d86f-5a5f-95cb-e8353a7e8356')
        super().__init__(uuid.UUID("b664a4a9-d86f-5a5f-95cb-e8353a7e8356"))
        self.run_options: Optional[RunOptions] = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        raise UnityCommunicationException(
            "The TrainingAnalyticsSideChannel received a message from Unity, "
            + "this should not have happened."
        )

    def environment_initialized(self, run_options: RunOptions) -> None:
        self.run_options = run_options
        # Tuple of (major, minor, patch)
        vi = sys.version_info
        env_params = run_options.environment_parameters

        msg = TrainingEnvironmentInitialized(
            python_version=f"{vi[0]}.{vi[1]}.{vi[2]}",
            mlagents_version=mlagents.trainers.__version__,
            mlagents_envs_version=mlagents_envs.__version__,
            torch_version=torch_utils.torch.__version__,
            torch_device_type=torch_utils.default_device().type,
            num_envs=run_options.env_settings.num_envs,
            num_environment_parameters=len(env_params) if env_params else 0,
        )

        any_message = Any()
        any_message.Pack(msg)

        env_init_msg = OutgoingMessage()
        env_init_msg.set_raw_bytes(any_message.SerializeToString())
        super().queue_message_to_send(env_init_msg)

    def training_started(self, behavior_name: str, config: TrainerSettings) -> None:
        msg = TrainingBehaviorInitialized(
            behavior_name=behavior_name,
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

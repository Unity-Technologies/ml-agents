# flake8: noqa

# TODO re-enable flake8
import sys
import uuid
import mlagents_envs
import mlagents.trainers
from mlagents import torch_utils
from mlagents.trainers.settings import RewardSignalType
from mlagents_envs.exception import UnityCommunicationException
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage

from mlagents.trainers.settings import TrainerSettings


class TrainingAnalyticsSideChannel(SideChannel):
    """
    Side channel that sends information about the training to the Unity environment so it can be logged.
    """

    def __init__(self) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/TrainingAnalyticsSideChannel")
        # UUID('b664a4a9-d86f-5a5f-95cb-e8353a7e8356')
        super().__init__(uuid.UUID("b664a4a9-d86f-5a5f-95cb-e8353a7e8356"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        raise UnityCommunicationException(
            "The TrainingAnalyticsSideChannel received a message from Unity, "
            + "this should not have happend."
        )

    def environment_initialized(self) -> None:
        # Tuple of (major, minor, patch)
        vi = sys.version_info
        python_version = f"{vi[0]}.{vi[1]}.{vi[2]}"
        mlagents_version = mlagents.trainers.__version__
        mlagents_envs_version = mlagents_envs.__version__
        torch_version = torch_utils.torch.__version__
        torch_device_type = torch_utils.default_device().type

        env_init_msg = OutgoingMessage()
        env_init_msg.write_string("environment_initialized!")
        super().queue_message_to_send(env_init_msg)

    def training_started(self, behavior_name: str, config: TrainerSettings) -> None:
        trainer_type = config.trainer_type

        extrinsic_reward_enabled = RewardSignalType.EXTRINSIC in config.reward_signals
        gail_reward_enabled = RewardSignalType.GAIL in config.reward_signals
        curiosity_reward_enabled = RewardSignalType.CURIOSITY in config.reward_signals
        rnd_reward_enabled = RewardSignalType.RND in config.reward_signals

        behavioral_cloning_enabled = config.behavioral_cloning is not None

        recurrent_enabled = config.network_settings.memory is not None
        visual_encoder = config.network_settings.vis_encode_type.value
        network_layers = config.network_settings.num_layers
        network_hidden_units = config.network_settings.hidden_units

        trainer_threaded = config.threaded
        self_play = config.self_play is not None

        training_start_msg = OutgoingMessage()
        training_start_msg.write_string(
            f"training_started for {behavior_name} with config {str(config)}!"
        )
        super().queue_message_to_send(training_start_msg)

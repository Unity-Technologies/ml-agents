import sys
import uuid
import mlagents_envs

from mlagents_envs.exception import UnityCommunicationException
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.communicator_objects.training_analytics_pb2 import (
    TrainingEnvironmentInitialized,
)
from google.protobuf.any_pb2 import Any


class DefaultTrainingAnalyticsSideChannel(SideChannel):
    """
    Side channel that sends information about the training to the Unity environment so it can be logged.
    """

    CHANNEL_ID = uuid.UUID("b664a4a9-d86f-5a5f-95cb-e8353a7e8356")

    def __init__(self) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/TrainingAnalyticsSideChannel")
        # UUID('b664a4a9-d86f-5a5f-95cb-e8353a7e8356')
        # We purposefully use the SAME side channel as the TrainingAnalyticsSideChannel

        super().__init__(DefaultTrainingAnalyticsSideChannel.CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:
        raise UnityCommunicationException(
            "The DefaultTrainingAnalyticsSideChannel received a message from Unity, "
            + "this should not have happened."
        )

    def environment_initialized(self) -> None:
        # Tuple of (major, minor, patch)
        vi = sys.version_info

        msg = TrainingEnvironmentInitialized(
            python_version=f"{vi[0]}.{vi[1]}.{vi[2]}",
            mlagents_version="Custom",
            mlagents_envs_version=mlagents_envs.__version__,
            torch_version="Unknown",
            torch_device_type="Unknown",
        )
        any_message = Any()
        any_message.Pack(msg)

        env_init_msg = OutgoingMessage()
        env_init_msg.set_raw_bytes(any_message.SerializeToString())  # type: ignore
        super().queue_message_to_send(env_init_msg)

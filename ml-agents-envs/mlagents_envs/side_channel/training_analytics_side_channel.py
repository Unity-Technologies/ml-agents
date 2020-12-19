import uuid
from mlagents_envs.exception import UnityCommunicationException
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage


class TrainingAnalyticsSideChannel(SideChannel):
    """
    Side channel that sends information about the training to the Unity environment so it can be logged.
    """

    def __init__(self) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/TrainingAnalyticsSideChannel")
        # UUID('b664a4a9-d86f-5a5f-95cb-e8353a7e8356')
        super().__init__(uuid.UUID("b664a4a9-d86f-5a5f-95cb-e8353a7e8356"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        Note that Python should never receive an engine configuration from
        Unity
        """
        raise UnityCommunicationException(
            "The EngineConfigurationChannel received a message from Unity, "
            + "this should not have happend."
        )

    def environment_initialized(self):
        env_init_msg = OutgoingMessage()
        env_init_msg.write_string("environment_initialized!")
        super().queue_message_to_send(env_init_msg)

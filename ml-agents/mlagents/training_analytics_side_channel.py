import uuid
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
        env_init_msg = OutgoingMessage()
        env_init_msg.write_string("environment_initialized!")
        super().queue_message_to_send(env_init_msg)

    def training_started(self, behavior_name: str, config: TrainerSettings) -> None:
        training_start_msg = OutgoingMessage()
        training_start_msg.write_string(
            f"training_started for {behavior_name} with config {str(config)}!"
        )
        super().queue_message_to_send(training_start_msg)

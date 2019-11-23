from mlagents.envs.side_channel.side_channel import SideChannel, ChannelType
from mlagents.envs.exception import UnityCommunicationException
import struct


class EngineConfigurationChannel(SideChannel):
    """
    This is  the SideChannel for engine configuration exchange. The data in the
    engine configuration is as follows :
     - int width;
     - int height;
     - int qualityLevel;
     - float timeScale;
     - int targetFrameRate;
    """

    def __init__(self):
        self.received_messages = []
        super().__init__()

    @property
    def channel_type(self) -> ChannelType:
        return 1

    def on_message_received(self, data: bytearray) -> None:
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

    def set_configuration(
        self,
        width: int = 80,
        height: int = 80,
        quality_level: int = 1,
        time_scale: float = 20.0,
        target_frame_rate: int = -1,
    ) -> None:
        """
        Sets the engine configuration. Takes as input the configurations of the
        engine.
        """
        data = bytearray()
        data += struct.pack("i", width)
        data += struct.pack("i", height)
        data += struct.pack("i", quality_level)
        data += struct.pack("f", time_scale)
        data += struct.pack("i", target_frame_rate)
        super().queue_message_to_send(data)

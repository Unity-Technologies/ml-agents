from mlagents_envs.side_channel.side_channel import SideChannel, SideChannelType
from mlagents_envs.exception import UnityCommunicationException
import struct
from typing import NamedTuple


class EngineConfig(NamedTuple):
    width: int
    height: int
    quality_level: int
    time_scale: float
    target_frame_rate: int

    @staticmethod
    def default_config():
        return EngineConfig(80, 80, 1, 20.0, -1)


class EngineConfigurationChannel(SideChannel):
    """
    This is the SideChannel for engine configuration exchange. The data in the
    engine configuration is as follows :
     - int width;
     - int height;
     - int qualityLevel;
     - float timeScale;
     - int targetFrameRate;
    """

    @property
    def channel_type(self) -> int:
        return SideChannelType.EngineSettings

    def on_message_received(self, data: bytes) -> None:
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

    def set_configuration_parameters(
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
        :param width: Defines the width of the display. Default 80.
        :param height: Defines the height of the display. Default 80.
        :param quality_level: Defines the quality level of the simulation.
        Default 1.
        :param time_scale: Defines the multiplier for the deltatime in the
        simulation. If set to a higher value, time will pass faster in the
        simulation but the physics might break. Default 20.
        :param target_frame_rate: Instructs simulation to try to render at a
        specified frame rate. Default -1.
        """
        data = bytearray()
        data += struct.pack("<i", width)
        data += struct.pack("<i", height)
        data += struct.pack("<i", quality_level)
        data += struct.pack("<f", time_scale)
        data += struct.pack("<i", target_frame_rate)
        super().queue_message_to_send(data)

    def set_configuration(self, config: EngineConfig) -> None:
        """
        Sets the engine configuration. Takes as input an EngineConfig.
        """
        data = bytearray()
        data += struct.pack("<iiifi", *config)
        super().queue_message_to_send(data)

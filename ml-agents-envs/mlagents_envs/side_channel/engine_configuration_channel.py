from mlagents_envs.side_channel import SideChannel, OutgoingMessage, IncomingMessage
from mlagents_envs.exception import UnityCommunicationException
import uuid
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

    def __init__(self) -> None:
        super().__init__(uuid.UUID("e951342c-4f7e-11ea-b238-784f4387d1f7"))

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
        msg = OutgoingMessage()
        msg.write_int32(width)
        msg.write_int32(height)
        msg.write_int32(quality_level)
        msg.write_float32(time_scale)
        msg.write_int32(target_frame_rate)
        super().queue_message_to_send(msg)

    def set_configuration(self, config: EngineConfig) -> None:
        """
        Sets the engine configuration. Takes as input an EngineConfig.
        """
        self.set_configuration_parameters(**config._asdict())

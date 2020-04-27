from mlagents_envs.side_channel import SideChannel, OutgoingMessage, IncomingMessage
from mlagents_envs.exception import (
    UnityCommunicationException,
    UnitySideChannelException,
)
import uuid
from typing import NamedTuple, Optional
from enum import IntEnum


class EngineConfig(NamedTuple):
    width: Optional[int]
    height: Optional[int]
    quality_level: Optional[int]
    time_scale: Optional[float]
    target_frame_rate: Optional[int]
    capture_frame_rate: Optional[int]

    @staticmethod
    def default_config():
        return EngineConfig(80, 80, 1, 20.0, -1, 60)


class EngineConfigurationChannel(SideChannel):
    """
    This is the SideChannel for engine configuration exchange. The data in the
    engine configuration is as follows :
     - int width;
     - int height;
     - int qualityLevel;
     - float timeScale;
     - int targetFrameRate;
     - int captureFrameRate;
    """

    class ConfigurationType(IntEnum):
        SCREEN_RESOLUTION = 0
        QUALITY_LEVEL = 1
        TIME_SCALE = 2
        TARGET_FRAME_RATE = 3
        CAPTURE_FRAME_RATE = 4

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
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality_level: Optional[int] = None,
        time_scale: Optional[float] = None,
        target_frame_rate: Optional[int] = None,
        capture_frame_rate: Optional[int] = None,
    ) -> None:
        """
        Sets the engine configuration. Takes as input the configurations of the
        engine.
        :param width: Defines the width of the display. (Must be set alongside height)
        :param height: Defines the height of the display. (Must be set alongside width)
        :param quality_level: Defines the quality level of the simulation.
        :param time_scale: Defines the multiplier for the deltatime in the
        simulation. If set to a higher value, time will pass faster in the
        simulation but the physics might break.
        :param target_frame_rate: Instructs simulation to try to render at a
        specified frame rate.
        :param capture_frame_rate: Instructs the simulation to consider time between
        updates to always be constant, regardless of the actual frame rate.
        """

        if (width is None and height is not None) or (
            width is not None and height is None
        ):
            raise UnitySideChannelException(
                "You cannot set the width/height of the screen resolution without also setting the height/width"
            )

        if width is not None and height is not None:
            screen_msg = OutgoingMessage()
            screen_msg.write_int32(self.ConfigurationType.SCREEN_RESOLUTION)
            screen_msg.write_int32(width)
            screen_msg.write_int32(height)
            super().queue_message_to_send(screen_msg)

        if quality_level is not None:
            quality_level_msg = OutgoingMessage()
            quality_level_msg.write_int32(self.ConfigurationType.QUALITY_LEVEL)
            quality_level_msg.write_int32(quality_level)
            super().queue_message_to_send(quality_level_msg)

        if time_scale is not None:
            time_scale_msg = OutgoingMessage()
            time_scale_msg.write_int32(self.ConfigurationType.TIME_SCALE)
            time_scale_msg.write_float32(time_scale)
            super().queue_message_to_send(time_scale_msg)

        if target_frame_rate is not None:
            target_frame_rate_msg = OutgoingMessage()
            target_frame_rate_msg.write_int32(self.ConfigurationType.TARGET_FRAME_RATE)
            target_frame_rate_msg.write_int32(target_frame_rate)
            super().queue_message_to_send(target_frame_rate_msg)

        if capture_frame_rate is not None:
            capture_frame_rate_msg = OutgoingMessage()
            capture_frame_rate_msg.write_int32(
                self.ConfigurationType.CAPTURE_FRAME_RATE
            )
            capture_frame_rate_msg.write_int32(capture_frame_rate)
            super().queue_message_to_send(capture_frame_rate_msg)

    def set_configuration(self, config: EngineConfig) -> None:
        """
        Sets the engine configuration. Takes as input an EngineConfig.
        """
        self.set_configuration_parameters(**config._asdict())

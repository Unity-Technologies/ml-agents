from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.exception import UnityCommunicationException
import uuid
from enum import IntEnum
from typing import List, Tuple


class EnvironmentParametersChannel(SideChannel):
    """
    This is the SideChannel for sending environment parameters to Unity.
    You can send parameters to an environment with the command
    set_float_parameter.
    """

    class EnvironmentDataTypes(IntEnum):
        FLOAT = 0
        SAMPLER = 1

    class SamplerTypes(IntEnum):
        UNIFORM = 0
        GAUSSIAN = 1
        MULTIRANGEUNIFORM = 2

    def __init__(self) -> None:
        channel_id = uuid.UUID("534c891e-810f-11ea-a9d0-822485860400")
        super().__init__(channel_id)

    def on_message_received(self, msg: IncomingMessage) -> None:
        raise UnityCommunicationException(
            "The EnvironmentParametersChannel received a message from Unity, "
            + "this should not have happend."
        )

    def set_float_parameter(self, key: str, value: float) -> None:
        """
        Sets a float environment parameter in the Unity Environment.
        :param key: The string identifier of the parameter.
        :param value: The float value of the parameter.
        """
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_int32(self.EnvironmentDataTypes.FLOAT)
        msg.write_float32(value)
        super().queue_message_to_send(msg)

    def set_uniform_sampler_parameters(
        self, key: str, min_value: float, max_value: float, seed: int
    ) -> None:
        """
        Sets a uniform environment parameter sampler.
        :param key: The string identifier of the parameter.
        :param min_value: The minimum of the sampling distribution.
        :param max_value: The maximum of the sampling distribution.
        :param seed: The random seed to initialize the sampler.
        """
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_int32(self.EnvironmentDataTypes.SAMPLER)
        msg.write_int32(seed)
        msg.write_int32(self.SamplerTypes.UNIFORM)
        msg.write_float32(min_value)
        msg.write_float32(max_value)
        super().queue_message_to_send(msg)

    def set_gaussian_sampler_parameters(
        self, key: str, mean: float, st_dev: float, seed: int
    ) -> None:
        """
        Sets a gaussian environment parameter sampler.
        :param key: The string identifier of the parameter.
        :param mean: The mean of the sampling distribution.
        :param st_dev: The standard deviation of the sampling distribution.
        :param seed: The random seed to initialize the sampler.
        """
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_int32(self.EnvironmentDataTypes.SAMPLER)
        msg.write_int32(seed)
        msg.write_int32(self.SamplerTypes.GAUSSIAN)
        msg.write_float32(mean)
        msg.write_float32(st_dev)
        super().queue_message_to_send(msg)

    def set_multirangeuniform_sampler_parameters(
        self, key: str, intervals: List[Tuple[float, float]], seed: int
    ) -> None:
        """
        Sets a multirangeuniform environment parameter sampler.
        :param key: The string identifier of the parameter.
        :param intervals: The lists of min and max that define each uniform distribution.
        :param seed: The random seed to initialize the sampler.
        """
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_int32(self.EnvironmentDataTypes.SAMPLER)
        msg.write_int32(seed)
        msg.write_int32(self.SamplerTypes.MULTIRANGEUNIFORM)
        flattened_intervals = [value for interval in intervals for value in interval]
        msg.write_float32_list(flattened_intervals)
        super().queue_message_to_send(msg)

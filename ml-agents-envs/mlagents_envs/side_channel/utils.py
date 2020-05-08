import uuid
import struct
from typing import Dict, Optional, List
from mlagents_envs.side_channel import SideChannel, IncomingMessage
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import get_logger


def parse_side_channel_message(
    side_channels: Dict[uuid.UUID, SideChannel], data: bytes
) -> None:
    offset = 0
    while offset < len(data):
        try:
            channel_id = uuid.UUID(bytes_le=bytes(data[offset : offset + 16]))
            offset += 16
            message_len, = struct.unpack_from("<i", data, offset)
            offset = offset + 4
            message_data = data[offset : offset + message_len]
            offset = offset + message_len
        except Exception:
            raise UnityEnvironmentException(
                "There was a problem reading a message in a SideChannel. "
                "Please make sure the version of MLAgents in Unity is "
                "compatible with the Python version."
            )
        if len(message_data) != message_len:
            raise UnityEnvironmentException(
                "The message received by the side channel {0} was "
                "unexpectedly short. Make sure your Unity Environment "
                "sending side channel data properly.".format(channel_id)
            )
        if channel_id in side_channels:
            incoming_message = IncomingMessage(message_data)
            side_channels[channel_id].on_message_received(incoming_message)
        else:
            get_logger(__name__).warning(
                "Unknown side channel data received. Channel type "
                ": {0}.".format(channel_id)
            )


def generate_side_channel_data(
    side_channels: Dict[uuid.UUID, SideChannel]
) -> bytearray:
    result = bytearray()
    for channel_id, channel in side_channels.items():
        for message in channel.message_queue:
            result += channel_id.bytes_le
            result += struct.pack("<i", len(message))
            result += message
        channel.message_queue = []
    return result


def get_side_channels_dict(
    side_c: Optional[List[SideChannel]]
) -> Dict[uuid.UUID, SideChannel]:
    side_channels_dict: Dict[uuid.UUID, SideChannel] = {}
    if side_c is not None:
        for _sc in side_c:
            if _sc.channel_id in side_channels_dict:
                raise UnityEnvironmentException(
                    f"There cannot be two side channels with "
                    f"the same channel id {_sc.channel_id}."
                )
            side_channels_dict[_sc.channel_id] = _sc
    return side_channels_dict

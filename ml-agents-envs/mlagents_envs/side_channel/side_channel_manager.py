import uuid
import struct
from typing import Dict, Optional, List
from mlagents_envs.side_channel import SideChannel, IncomingMessage
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.logging_util import get_logger


class SideChannelManager:
    def __init__(self, side_channels=Optional[List[SideChannel]]):
        self._side_channels_dict = self._get_side_channels_dict(side_channels)

    def process_side_channel_message(self, data: bytes) -> None:
        """
        Separates the data received from Python into individual messages for each
        registered side channel and calls on_message_received on them.
        :param data: The packed message sent by Unity
        """
        offset = 0
        while offset < len(data):
            try:
                channel_id = uuid.UUID(bytes_le=bytes(data[offset : offset + 16]))
                offset += 16
                message_len, = struct.unpack_from("<i", data, offset)
                offset = offset + 4
                message_data = data[offset : offset + message_len]
                offset = offset + message_len
            except (struct.error, ValueError, IndexError):
                raise UnityEnvironmentException(
                    "There was a problem reading a message in a SideChannel. "
                    "Please make sure the version of MLAgents in Unity is "
                    "compatible with the Python version."
                )
            if len(message_data) != message_len:
                raise UnityEnvironmentException(
                    "The message received by the side channel {} was "
                    "unexpectedly short. Make sure your Unity Environment "
                    "sending side channel data properly.".format(channel_id)
                )
            if channel_id in self._side_channels_dict:
                incoming_message = IncomingMessage(message_data)
                self._side_channels_dict[channel_id].on_message_received(
                    incoming_message
                )
            else:
                get_logger(__name__).warning(
                    f"Unknown side channel data received. Channel type: {channel_id}."
                )

    def generate_side_channel_messages(self) -> bytearray:
        """
        Gathers the messages that the registered side channels will send to Unity
        and combines them into a single message ready to be sent.
        """
        result = bytearray()
        for channel_id, channel in self._side_channels_dict.items():
            for message in channel.message_queue:
                result += channel_id.bytes_le
                result += struct.pack("<i", len(message))
                result += message
            channel.message_queue = []
        return result

    @staticmethod
    def _get_side_channels_dict(
        side_channels: Optional[List[SideChannel]]
    ) -> Dict[uuid.UUID, SideChannel]:
        """
        Converts a list of side channels into a dictionary of channel_id to SideChannel
        :param side_channels: The list of side channels.
        """
        side_channels_dict: Dict[uuid.UUID, SideChannel] = {}
        if side_channels is not None:
            for _sc in side_channels:
                if _sc.channel_id in side_channels_dict:
                    raise UnityEnvironmentException(
                        f"There cannot be two side channels with "
                        f"the same channel id {_sc.channel_id}."
                    )
                side_channels_dict[_sc.channel_id] = _sc
        return side_channels_dict

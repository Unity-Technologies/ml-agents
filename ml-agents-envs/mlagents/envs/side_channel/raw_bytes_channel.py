from mlagents.envs.side_channel.side_channel import SideChannel, SideChannelType
from typing import List


class RawBytesChannel(SideChannel):
    """
    This is an example of what the SideChannel for raw bytes exchange would
    look like. Is meant to be used for general research purpose.
    """

    def __init__(self, channel_id=0):
        self._received_messages = []
        self._channel_id = channel_id
        super().__init__()

    @property
    def channel_type(self) -> int:
        return SideChannelType.RawBytesChannelStart + self._channel_id

    def on_message_received(self, data: bytearray) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        """
        self._received_messages.append(data)

    def receive_raw_bytes(self) -> List[bytearray]:
        """
        returns a list of bytearray received from the environment.
        """
        result = []
        for m in self._received_messages:
            result.append(m)
        self._received_messages = []
        return result

    def send_raw_data(self, data: bytearray) -> None:
        """
        Queues a message to be sent by the environment at the next call to
        step.
        """
        super().queue_message_to_send(data)

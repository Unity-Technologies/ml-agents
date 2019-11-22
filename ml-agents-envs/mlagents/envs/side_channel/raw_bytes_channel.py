import Queue
from mlagents.envs.side_channel import SideChannel, ChannelType
from typing import List


class RawBytesChannel(SideChannel):
    """
    This is an example of what the SideChannel for raw bytes exchange would
    look like. Is meant to be used for general research purpose.
    """

    def __init__(self):
        self.received_messages = Queue.Queue()
        super().__init__()

    @property
    def channel_type(self) -> ChannelType:
        return 0

    def on_message_received(self, data: bytearray) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        """
        self.received_messages.put(data)

    def receive_raw_bytes(self) -> List[bytearray]:
        """
        returns a list of bytearray received from the environment.
        """
        result = []
        while not self.received_messages.empty():
            result.append(self.received_messages.get())
        return result

    def send_raw_data(self, data: bytearray) -> None:
        """
        Queues a message to be sent by the environment at the next call to
        step.
        """
        super().queue_message_to_send(data)

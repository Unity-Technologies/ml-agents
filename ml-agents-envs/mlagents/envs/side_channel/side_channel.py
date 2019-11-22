import Queue
from abc import ABC, abstractmethod

ChannelType = int


class SideChannel(ABC):
    """
    The side channel just get access to a bytes buffer that will be shared
    between C# and Python. For example, We will create a specific side channel
    for properties that will be a list of string (fixed size) to float number,
    that can be modified by both C# and Python. All side channels are passed
    to the Env object at construction.
    """

    def __init__(self):
        self.message_queue = Queue.Queue()

    def queue_message_to_send(self, data: bytearray) -> None:
        """
        Queues a message to be sent by the environment at the next call to
        step.
        """
        self.message_queue.put(data)

    @abstractmethod
    def on_message_received(self, data: bytearray) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        """
        pass

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """
        :return:The type of side channel used. Will influence how the data is
        processed in the environment.
        """
        pass

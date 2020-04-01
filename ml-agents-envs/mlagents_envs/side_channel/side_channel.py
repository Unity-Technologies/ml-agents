from abc import ABC, abstractmethod
from typing import List
import uuid

from mlagents_envs.side_channel import IncomingMessage, OutgoingMessage
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class SideChannel(ABC):
    """
    The side channel just get access to a bytes buffer that will be shared
    between C# and Python. For example, We will create a specific side channel
    for properties that will be a list of string (fixed size) to float number,
    that can be modified by both C# and Python. All side channels are passed
    to the Env object at construction.
    """

    def __init__(self, channel_id: uuid.UUID):
        self._channel_id: uuid.UUID = channel_id
        self.message_queue: List[bytearray] = []

    def queue_message_to_send(self, msg: OutgoingMessage) -> None:
        """
        Queues a message to be sent by the environment at the next call to
        step.
        """
        self.message_queue.append(msg.buffer)

    @abstractmethod
    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        """
        pass

    @property
    def channel_id(self) -> uuid.UUID:
        """
        :return:The type of side channel used. Will influence how the data is
        processed in the environment.
        """
        return self._channel_id

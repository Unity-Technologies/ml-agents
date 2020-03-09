from abc import ABC, abstractmethod
from typing import List
import uuid
import struct

import logging

logger = logging.getLogger(__name__)


class SideChannel(ABC):
    """
    The side channel just get access to a bytes buffer that will be shared
    between C# and Python. For example, We will create a specific side channel
    for properties that will be a list of string (fixed size) to float number,
    that can be modified by both C# and Python. All side channels are passed
    to the Env object at construction.
    """

    def __init__(self, channel_id):
        self._channel_id: uuid.UUID = channel_id
        self.message_queue: List[bytearray] = []

    def queue_message_to_send(self, msg: "OutgoingMessage") -> None:
        """
        Queues a message to be sent by the environment at the next call to
        step.
        """
        self.message_queue.append(msg.buffer)

    @abstractmethod
    def on_message_received(self, msg: "IncomingMessage") -> None:
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


class OutgoingMessage:
    def __init__(self):
        self.buffer = bytearray()

    def write_int32(self, i: int) -> None:
        self.buffer += struct.pack("<i", i)

    def write_float32(self, f: float) -> None:
        self.buffer += struct.pack("<f", f)

    def write_string(self, s: str) -> None:
        encoded_key = s.encode("ascii")
        self.write_int32(len(encoded_key))
        self.buffer += encoded_key

    def set_raw_bytes(self, buffer: bytearray) -> None:
        if self.buffer:
            logger.warning(
                "Called set_raw_bytes but the message already has been written to. This will overwrite data."
            )
        self.buffer = bytearray(buffer)


class IncomingMessage:
    def __init__(self, buffer: bytes, offset: int = 0):
        self.buffer = buffer
        self.offset = offset

    def read_int32(self) -> int:
        val = struct.unpack_from("<i", self.buffer, self.offset)[0]
        self.offset += 4
        return val

    def read_float32(self) -> float:
        val = struct.unpack_from("<f", self.buffer, self.offset)[0]
        self.offset += 4
        return val

    def read_string(self) -> str:
        encoded_str_len = self.read_int32()
        val = self.buffer[self.offset : self.offset + encoded_str_len].decode("ascii")
        self.offset += encoded_str_len
        return val

    def get_raw_bytes(self) -> bytes:
        return bytearray(self.buffer)

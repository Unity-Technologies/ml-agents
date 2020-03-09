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

    def __init__(self, channel_id: uuid.UUID):
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
    """
    Utility class for forming the message that is written to a SideChannel.
    All data is written in little-endian format using the struct module.
    """

    def __init__(self):
        """
        Create an OutgoingMessage with an empty buffer.
        """
        self.buffer = bytearray()

    def write_bool(self, b: bool) -> None:
        """
        Append a boolean value.
        """
        self.buffer += struct.pack("<?", b)

    def write_int32(self, i: int) -> None:
        """
        Append an integer value.
        """
        self.buffer += struct.pack("<i", i)

    def write_float32(self, f: float) -> None:
        """
        Append a float value. It will be truncated to 32-bit precision.
        """
        self.buffer += struct.pack("<f", f)

    def write_float32_list(self, float_list: List[float]) -> None:
        """
        Append a list of float values. They will be truncated to 32-bit precision.
        """
        self.write_int32(len(float_list))
        for f in float_list:
            self.write_float32(f)

    def write_string(self, s: str) -> None:
        """
        Append a string value. Internally, it will be encoded to ascii, and the
        encoded length will also be written to the message.
        """
        encoded_key = s.encode("ascii")
        self.write_int32(len(encoded_key))
        self.buffer += encoded_key

    def set_raw_bytes(self, buffer: bytearray) -> None:
        """
        Set the internal buffer to a new bytearray. This will overwrite any existing data.
        :param buffer:
        :return:
        """
        if self.buffer:
            logger.warning(
                "Called set_raw_bytes but the message already has been written to. This will overwrite data."
            )
        self.buffer = bytearray(buffer)


class IncomingMessage:
    """
    Utility class for reading the message written to a SideChannel.
    Values must be read in the order they were written.
    """

    def __init__(self, buffer: bytes, offset: int = 0):
        """
        Create a new IncomingMessage from the bytes.
        """
        self.buffer = buffer
        self.offset = offset

    def read_bool(self) -> bool:
        """
        Read a boolean value from the message buffer.
        """
        val = struct.unpack_from("<?", self.buffer, self.offset)[0]
        self.offset += 1
        return val

    def read_int32(self) -> int:
        """
        Read an integer value from the message buffer.
        """
        val = struct.unpack_from("<i", self.buffer, self.offset)[0]
        self.offset += 4
        return val

    def read_float32(self) -> float:
        """
        Read a float value from the message buffer.
        """
        val = struct.unpack_from("<f", self.buffer, self.offset)[0]
        self.offset += 4
        return val

    def read_float32_list(self) -> List[float]:
        """
        Read a list of float values from the message buffer.
        """
        list_len = self.read_int32()
        output = []
        for _ in range(list_len):
            output.append(self.read_float32())
        return output

    def read_string(self) -> str:
        """
        Read a string value from the message buffer.
        """
        encoded_str_len = self.read_int32()
        val = self.buffer[self.offset : self.offset + encoded_str_len].decode("ascii")
        self.offset += encoded_str_len
        return val

    def get_raw_bytes(self) -> bytes:
        """
        Get a copy of the internal bytes used by the message.
        """
        return bytearray(self.buffer)

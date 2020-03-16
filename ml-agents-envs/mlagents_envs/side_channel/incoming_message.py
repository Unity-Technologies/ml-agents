from typing import List
import struct


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

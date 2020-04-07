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

    def read_bool(self, default_value: bool = False) -> bool:
        """
        Read a boolean value from the message buffer.
        :param default_value: Default value to use if the end of the message is reached.
        :return: The value read from the message, or the default value if the end was reached.
        """
        if self._at_end_of_buffer():
            return default_value

        val = struct.unpack_from("<?", self.buffer, self.offset)[0]
        self.offset += 1
        return val

    def read_int32(self, default_value: int = 0) -> int:
        """
        Read an integer value from the message buffer.
        :param default_value: Default value to use if the end of the message is reached.
        :return: The value read from the message, or the default value if the end was reached.
        """
        if self._at_end_of_buffer():
            return default_value

        val = struct.unpack_from("<i", self.buffer, self.offset)[0]
        self.offset += 4
        return val

    def read_float32(self, default_value: float = 0.0) -> float:
        """
        Read a float value from the message buffer.
        :param default_value: Default value to use if the end of the message is reached.
        :return: The value read from the message, or the default value if the end was reached.
        """
        if self._at_end_of_buffer():
            return default_value

        val = struct.unpack_from("<f", self.buffer, self.offset)[0]
        self.offset += 4
        return val

    def read_float32_list(self, default_value: List[float] = None) -> List[float]:
        """
        Read a list of float values from the message buffer.
        :param default_value: Default value to use if the end of the message is reached.
        :return: The value read from the message, or the default value if the end was reached.
        """
        if self._at_end_of_buffer():
            return [] if default_value is None else default_value

        list_len = self.read_int32()
        output = []
        for _ in range(list_len):
            output.append(self.read_float32())
        return output

    def read_string(self, default_value: str = "") -> str:
        """
        Read a string value from the message buffer.
        :param default_value: Default value to use if the end of the message is reached.
        :return: The value read from the message, or the default value if the end was reached.
        """
        if self._at_end_of_buffer():
            return default_value

        encoded_str_len = self.read_int32()
        val = self.buffer[self.offset : self.offset + encoded_str_len].decode("ascii")
        self.offset += encoded_str_len
        return val

    def get_raw_bytes(self) -> bytes:
        """
        Get a copy of the internal bytes used by the message.
        """
        return bytearray(self.buffer)

    def _at_end_of_buffer(self) -> bool:
        return self.offset >= len(self.buffer)

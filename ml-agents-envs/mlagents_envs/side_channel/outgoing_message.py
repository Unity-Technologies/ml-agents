from typing import List
import struct

import logging

logger = logging.getLogger(__name__)


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

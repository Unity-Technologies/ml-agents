from mlagents.envs.side_channel.side_channel import SideChannel, SideChannelType
import struct
from typing import Tuple, Optional, List


class FloatPropertiesChannel(SideChannel):
    """
    This is the SideChannel for float properties shared with Unity.
    You can modify the float properties of an environment with the commands
    set_property, get_property and list_properties.
    """

    def __init__(self):
        self._float_properties = {}
        super().__init__()

    @property
    def channel_type(self) -> int:
        return SideChannelType.FloatProperties

    def on_message_received(self, data: bytearray) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        Note that Python should never receive an engine configuration from
        Unity
        """
        k, v = self.deserialize_float_prop(data)
        self._float_properties[k] = v

    def set_property(self, key: str, value: float) -> None:
        self._float_properties[key] = value
        super().queue_message_to_send(self.serialize_float_prop(key, value))

    def get_property(self, key: str) -> Optional[float]:
        if key in self._float_properties:
            return self._float_properties[key]
        else:
            return None

    def list_properties(self) -> List[str]:
        return self._float_properties.keys()

    @staticmethod
    def serialize_float_prop(key: str, value: float) -> bytearray:
        result = bytearray()
        encoded_key = key.encode("ascii")
        result += struct.pack("<i", len(encoded_key))
        result += encoded_key
        result += struct.pack("<f", value)
        return result

    @staticmethod
    def deserialize_float_prop(data: bytearray) -> Tuple[str, float]:
        offset = 0
        encoded_key_len = struct.unpack_from("<i", data, offset)[0]
        offset = offset + 4
        key = data[offset : offset + encoded_key_len].decode("ascii")
        offset = offset + encoded_key_len
        value = struct.unpack_from("<f", data, offset)[0]
        return key, value

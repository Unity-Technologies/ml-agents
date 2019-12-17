from mlagents_envs.side_channel.side_channel import SideChannel, SideChannelType
import struct
from typing import Dict, Tuple, Optional, List


class FloatPropertiesChannel(SideChannel):
    """
    This is the SideChannel for float properties shared with Unity.
    You can modify the float properties of an environment with the commands
    set_property, get_property and list_properties.
    """

    def __init__(self) -> None:
        self._float_properties: Dict[str, float] = {}
        super().__init__()

    @property
    def channel_type(self) -> int:
        return SideChannelType.FloatProperties

    def on_message_received(self, data: bytes) -> None:
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
        """
        Sets a property in the Unity Environment.
        :param key: The string identifier of the property.
        :param value: The float value of the property.
        """
        self._float_properties[key] = value
        super().queue_message_to_send(self.serialize_float_prop(key, value))

    def get_property(self, key: str) -> Optional[float]:
        """
        Gets a property in the Unity Environment. If the property was not
        found, will return None.
        :param key: The string identifier of the property.
        :return: The float value of the property or None.
        """
        return self._float_properties.get(key)

    def list_properties(self) -> List[str]:
        """
        Returns a list of all the string identifiers of the properties
        currently present in the Unity Environment.
        """
        return list(self._float_properties.keys())

    def get_property_dict_copy(self) -> Dict[str, float]:
        """
        Returns a copy of the float properties.
        :return:
        """
        return dict(self._float_properties)

    @staticmethod
    def serialize_float_prop(key: str, value: float) -> bytearray:
        result = bytearray()
        encoded_key = key.encode("ascii")
        result += struct.pack("<i", len(encoded_key))
        result += encoded_key
        result += struct.pack("<f", value)
        return result

    @staticmethod
    def deserialize_float_prop(data: bytes) -> Tuple[str, float]:
        offset = 0
        encoded_key_len = struct.unpack_from("<i", data, offset)[0]
        offset = offset + 4
        key = data[offset : offset + encoded_key_len].decode("ascii")
        offset = offset + encoded_key_len
        value = struct.unpack_from("<f", data, offset)[0]
        return key, value

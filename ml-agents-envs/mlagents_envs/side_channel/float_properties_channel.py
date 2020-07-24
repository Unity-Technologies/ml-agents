from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
import uuid
from typing import Dict, Optional, List


class FloatPropertiesChannel(SideChannel):
    """
    This is the SideChannel for float properties shared with Unity.
    You can modify the float properties of an environment with the commands
    set_property, get_property and list_properties.
    """

    def __init__(self, channel_id: uuid.UUID = None) -> None:
        self._float_properties: Dict[str, float] = {}
        if channel_id is None:
            channel_id = uuid.UUID("60ccf7d0-4f7e-11ea-b238-784f4387d1f7")
        super().__init__(channel_id)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Is called by the environment to the side channel. Can be called
        multiple times per step if multiple messages are meant for that
        SideChannel.
        """
        k = msg.read_string()
        v = msg.read_float32()
        self._float_properties[k] = v

    def set_property(self, key: str, value: float) -> None:
        """
        Sets a property in the Unity Environment.
        :param key: The string identifier of the property.
        :param value: The float value of the property.
        """
        self._float_properties[key] = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32(value)
        super().queue_message_to_send(msg)

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

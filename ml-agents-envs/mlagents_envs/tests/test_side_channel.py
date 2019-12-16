import struct
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.environment import UnityEnvironment


class IntChannel(SideChannel):
    def __init__(self):
        self.list_int = []
        super().__init__()

    @property
    def channel_type(self):
        return -1

    def on_message_received(self, data):
        val = struct.unpack_from("<i", data, 0)[0]
        self.list_int += [val]

    def send_int(self, value):
        data = bytearray()
        data += struct.pack("<i", value)
        super().queue_message_to_send(data)


def test_int_channel():
    sender = IntChannel()
    receiver = IntChannel()
    sender.send_int(5)
    sender.send_int(6)
    data = UnityEnvironment._generate_side_channel_data({sender.channel_type: sender})
    UnityEnvironment._parse_side_channel_message(
        {receiver.channel_type: receiver}, data
    )
    assert receiver.list_int[0] == 5
    assert receiver.list_int[1] == 6


def test_float_properties():
    sender = FloatPropertiesChannel()
    receiver = FloatPropertiesChannel()

    sender.set_property("prop1", 1.0)

    data = UnityEnvironment._generate_side_channel_data({sender.channel_type: sender})
    UnityEnvironment._parse_side_channel_message(
        {receiver.channel_type: receiver}, data
    )

    val = receiver.get_property("prop1")
    assert val == 1.0
    val = receiver.get_property("prop2")
    assert val is None
    sender.set_property("prop2", 2.0)

    data = UnityEnvironment._generate_side_channel_data({sender.channel_type: sender})
    UnityEnvironment._parse_side_channel_message(
        {receiver.channel_type: receiver}, data
    )

    val = receiver.get_property("prop1")
    assert val == 1.0
    val = receiver.get_property("prop2")
    assert val == 2.0
    assert len(receiver.list_properties()) == 2
    assert "prop1" in receiver.list_properties()
    assert "prop2" in receiver.list_properties()
    val = sender.get_property("prop1")
    assert val == 1.0

    assert receiver.get_property_dict_copy() == {"prop1": 1.0, "prop2": 2.0}
    assert receiver.get_property_dict_copy() == sender.get_property_dict_copy()


def test_raw_bytes():
    sender = RawBytesChannel()
    receiver = RawBytesChannel()

    sender.send_raw_data("foo".encode("ascii"))
    sender.send_raw_data("bar".encode("ascii"))

    data = UnityEnvironment._generate_side_channel_data({sender.channel_type: sender})
    UnityEnvironment._parse_side_channel_message(
        {receiver.channel_type: receiver}, data
    )

    messages = receiver.get_and_clear_received_messages()
    assert len(messages) == 2
    assert messages[0].decode("ascii") == "foo"
    assert messages[1].decode("ascii") == "bar"

    messages = receiver.get_and_clear_received_messages()
    assert len(messages) == 0

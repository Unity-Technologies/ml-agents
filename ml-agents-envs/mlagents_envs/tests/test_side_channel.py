import uuid
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.environment import UnityEnvironment


class IntChannel(SideChannel):
    def __init__(self):
        self.list_int = []
        super().__init__(uuid.UUID("a85ba5c0-4f87-11ea-a517-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        val = msg.read_int32()
        self.list_int += [val]

    def send_int(self, value):
        msg = OutgoingMessage()
        msg.write_int32(value)
        super().queue_message_to_send(msg)


def test_int_channel():
    sender = IntChannel()
    receiver = IntChannel()
    sender.send_int(5)
    sender.send_int(6)
    data = UnityEnvironment._generate_side_channel_data({sender.channel_id: sender})
    UnityEnvironment._parse_side_channel_message({receiver.channel_id: receiver}, data)
    assert receiver.list_int[0] == 5
    assert receiver.list_int[1] == 6


def test_float_properties():
    sender = FloatPropertiesChannel()
    receiver = FloatPropertiesChannel()

    sender.set_property("prop1", 1.0)

    data = UnityEnvironment._generate_side_channel_data({sender.channel_id: sender})
    UnityEnvironment._parse_side_channel_message({receiver.channel_id: receiver}, data)

    val = receiver.get_property("prop1")
    assert val == 1.0
    val = receiver.get_property("prop2")
    assert val is None
    sender.set_property("prop2", 2.0)

    data = UnityEnvironment._generate_side_channel_data({sender.channel_id: sender})
    UnityEnvironment._parse_side_channel_message({receiver.channel_id: receiver}, data)

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
    guid = uuid.uuid4()
    sender = RawBytesChannel(guid)
    receiver = RawBytesChannel(guid)

    sender.send_raw_data("foo".encode("ascii"))
    sender.send_raw_data("bar".encode("ascii"))

    data = UnityEnvironment._generate_side_channel_data({sender.channel_id: sender})
    UnityEnvironment._parse_side_channel_message({receiver.channel_id: receiver}, data)

    messages = receiver.get_and_clear_received_messages()
    assert len(messages) == 2
    assert messages[0].decode("ascii") == "foo"
    assert messages[1].decode("ascii") == "bar"

    messages = receiver.get_and_clear_received_messages()
    assert len(messages) == 0


def test_message_bool():
    vals = [True, False]
    msg_out = OutgoingMessage()
    for v in vals:
        msg_out.write_bool(v)

    msg_in = IncomingMessage(msg_out.buffer)
    read_vals = []
    for _ in range(len(vals)):
        read_vals.append(msg_in.read_bool())
    assert vals == read_vals


def test_message_int32():
    val = 1337
    msg_out = OutgoingMessage()
    msg_out.write_int32(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_int32()
    assert val == read_val


def test_message_float32():
    val = 42.0
    msg_out = OutgoingMessage()
    msg_out.write_float32(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_float32()
    # These won't be exactly equal in general, since python floats are 64-bit.
    assert val == read_val


def test_message_string():
    val = "mlagents!"
    msg_out = OutgoingMessage()
    msg_out.write_string(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_string()
    assert val == read_val


def test_message_float_list():
    val = [1.0, 3.0, 9.0]
    msg_out = OutgoingMessage()
    msg_out.write_float32_list(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_float32_list()
    # These won't be exactly equal in general, since python floats are 64-bit.
    assert val == read_val

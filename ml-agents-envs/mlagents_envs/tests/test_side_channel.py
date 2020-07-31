import uuid
import pytest
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.side_channel.side_channel_manager import SideChannelManager
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
    EngineConfig,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.stats_side_channel import (
    StatsSideChannel,
    StatsAggregationMethod,
)
from mlagents_envs.exception import (
    UnitySideChannelException,
    UnityCommunicationException,
)


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
    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)
    assert receiver.list_int[0] == 5
    assert receiver.list_int[1] == 6


def test_float_properties():
    sender = FloatPropertiesChannel()
    receiver = FloatPropertiesChannel()

    sender.set_property("prop1", 1.0)

    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

    val = receiver.get_property("prop1")
    assert val == 1.0
    val = receiver.get_property("prop2")
    assert val is None
    sender.set_property("prop2", 2.0)

    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

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

    sender.send_raw_data(b"foo")
    sender.send_raw_data(b"bar")

    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

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

    # Test reading with defaults
    assert msg_in.read_bool() is False
    assert msg_in.read_bool(default_value=True) is True


def test_message_int32():
    val = 1337
    msg_out = OutgoingMessage()
    msg_out.write_int32(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_int32()
    assert val == read_val

    # Test reading with defaults
    assert 0 == msg_in.read_int32()
    assert val == msg_in.read_int32(default_value=val)


def test_message_float32():
    val = 42.0
    msg_out = OutgoingMessage()
    msg_out.write_float32(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_float32()
    # These won't be exactly equal in general, since python floats are 64-bit.
    assert val == read_val

    # Test reading with defaults
    assert 0.0 == msg_in.read_float32()
    assert val == msg_in.read_float32(default_value=val)


def test_message_string():
    val = "mlagents!"
    msg_out = OutgoingMessage()
    msg_out.write_string(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_string()
    assert val == read_val

    # Test reading with defaults
    assert "" == msg_in.read_string()
    assert val == msg_in.read_string(default_value=val)


def test_message_float_list():
    val = [1.0, 3.0, 9.0]
    msg_out = OutgoingMessage()
    msg_out.write_float32_list(val)

    msg_in = IncomingMessage(msg_out.buffer)
    read_val = msg_in.read_float32_list()
    # These won't be exactly equal in general, since python floats are 64-bit.
    assert val == read_val

    # Test reading with defaults
    assert [] == msg_in.read_float32_list()
    assert val == msg_in.read_float32_list(default_value=val)


def test_engine_configuration():
    sender = EngineConfigurationChannel()
    # We use a raw bytes channel to interpred the data
    receiver = RawBytesChannel(sender.channel_id)

    config = EngineConfig.default_config()
    sender.set_configuration(config)
    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

    received_data = receiver.get_and_clear_received_messages()
    assert len(received_data) == 5  # 5 different messages one for each setting

    sent_time_scale = 4.5
    sender.set_configuration_parameters(time_scale=sent_time_scale)

    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

    message = IncomingMessage(receiver.get_and_clear_received_messages()[0])
    message.read_int32()
    time_scale = message.read_float32()
    assert time_scale == sent_time_scale

    with pytest.raises(UnitySideChannelException):
        sender.set_configuration_parameters(width=None, height=42)

    with pytest.raises(UnityCommunicationException):
        # try to send data to the EngineConfigurationChannel
        sender.set_configuration_parameters(time_scale=sent_time_scale)
        data = SideChannelManager([sender]).generate_side_channel_messages()
        SideChannelManager([sender]).process_side_channel_message(data)


def test_environment_parameters():
    sender = EnvironmentParametersChannel()
    # We use a raw bytes channel to interpred the data
    receiver = RawBytesChannel(sender.channel_id)

    sender.set_float_parameter("param-1", 0.1)
    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

    message = IncomingMessage(receiver.get_and_clear_received_messages()[0])
    key = message.read_string()
    dtype = message.read_int32()
    value = message.read_float32()
    assert key == "param-1"
    assert dtype == EnvironmentParametersChannel.EnvironmentDataTypes.FLOAT
    assert value - 0.1 < 1e-8

    sender.set_float_parameter("param-1", 0.1)
    sender.set_float_parameter("param-2", 0.1)
    sender.set_float_parameter("param-3", 0.1)

    data = SideChannelManager([sender]).generate_side_channel_messages()
    SideChannelManager([receiver]).process_side_channel_message(data)

    assert len(receiver.get_and_clear_received_messages()) == 3

    with pytest.raises(UnityCommunicationException):
        # try to send data to the EngineConfigurationChannel
        sender.set_float_parameter("param-1", 0.1)
        data = SideChannelManager([sender]).generate_side_channel_messages()
        SideChannelManager([sender]).process_side_channel_message(data)


def test_stats_channel():
    receiver = StatsSideChannel()
    message = OutgoingMessage()
    message.write_string("stats-1")
    message.write_float32(42.0)
    message.write_int32(1)  # corresponds to StatsAggregationMethod.MOST_RECENT

    receiver.on_message_received(IncomingMessage(message.buffer))

    stats = receiver.get_and_reset_stats()

    assert len(stats) == 1
    val, method = stats["stats-1"][0]
    assert val - 42.0 < 1e-8
    assert method == StatsAggregationMethod.MOST_RECENT

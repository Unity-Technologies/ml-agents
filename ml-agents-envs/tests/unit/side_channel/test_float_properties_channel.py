from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.environment import UnityEnvironment


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

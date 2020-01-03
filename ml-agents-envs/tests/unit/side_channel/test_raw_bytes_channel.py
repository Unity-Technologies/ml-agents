from mlagents_envs.side_channel.raw_bytes_channel import RawBytesChannel
from mlagents_envs.environment import UnityEnvironment


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

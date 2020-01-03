import struct
from mlagents_envs.side_channel.side_channel import SideChannel
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

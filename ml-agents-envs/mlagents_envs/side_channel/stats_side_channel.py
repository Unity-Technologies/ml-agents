from mlagents_envs.side_channel import SideChannel, IncomingMessage
import uuid
from typing import Dict


class StatsSideChannel(SideChannel):
    def __init__(self) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/StatsSideChannel")
        # UUID('a1d8f7b7-cec8-50f9-b78b-d3e165a78520')
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

        self.stats: Dict[str, float] = {}

    def on_message_received(self, msg: IncomingMessage) -> None:
        key = msg.read_string()
        val = msg.read_float32()
        self.stats[key] = val

    def get_and_reset_stats(self) -> Dict[str, float]:
        s = self.stats
        self.stats = {}
        return s

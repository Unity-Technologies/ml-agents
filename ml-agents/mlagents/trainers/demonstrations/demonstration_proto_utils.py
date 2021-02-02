import os
from typing import List, Tuple
import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.rpc_utils import behavior_spec_from_proto, steps_from_proto
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.timers import timed, hierarchical_timer
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore
from google.protobuf.internal.encoder import _EncodeVarint  # type: ignore


INITIAL_POS = 33
SUPPORTED_DEMONSTRATION_VERSIONS = frozenset([0, 1])


@timed
def load_demonstration(
    file_paths: List[str],
) -> Tuple[BehaviorSpec, List[AgentInfoActionPairProto]]:
    """
    Loads and parses a demonstration file.
    :param file_path: Location of demonstration file (.demo).
    :return: BrainParameter and list of AgentInfoActionPairProto containing demonstration data.
    """

    # First 32 bytes of file dedicated to meta-data.
    behavior_spec = None
    brain_param_proto = None
    info_action_pairs = []
    total_expected = 0
    for _file_path in file_paths:
        with open(_file_path, "rb") as fp:
            with hierarchical_timer("read_file"):
                data = fp.read()
            next_pos, pos, obs_decoded = 0, 0, 0
            while pos < len(data):
                next_pos, pos = _DecodeVarint32(data, pos)
                if obs_decoded == 0:
                    meta_data_proto = DemonstrationMetaProto()
                    meta_data_proto.ParseFromString(data[pos : pos + next_pos])
                    if (
                        meta_data_proto.api_version
                        not in SUPPORTED_DEMONSTRATION_VERSIONS
                    ):
                        raise RuntimeError(
                            f"Can't load Demonstration data from an unsupported version ({meta_data_proto.api_version})"
                        )
                    total_expected += meta_data_proto.number_steps
                    pos = INITIAL_POS
                if obs_decoded == 1:
                    brain_param_proto = BrainParametersProto()
                    brain_param_proto.ParseFromString(data[pos : pos + next_pos])
                    pos += next_pos
                if obs_decoded > 1:
                    agent_info_action = AgentInfoActionPairProto()
                    agent_info_action.ParseFromString(data[pos : pos + next_pos])
                    if behavior_spec is None:
                        behavior_spec = behavior_spec_from_proto(
                            brain_param_proto, agent_info_action.agent_info
                        )
                    info_action_pairs.append(agent_info_action)
                    if len(info_action_pairs) == total_expected:
                        break
                    pos += next_pos
                obs_decoded += 1
    if not behavior_spec:
        raise RuntimeError(
            f"No BrainParameters found in demonstration file(s) at {file_paths}."
        )
    return behavior_spec, info_action_pairs


def write_delimited(f, message):
    msg_string = message.SerializeToString()
    msg_size = len(msg_string)
    _EncodeVarint(f.write, msg_size)
    f.write(msg_string)


def write_demo(demo_path, meta_data_proto, brain_param_proto, agent_info_protos):
    with open(demo_path, "wb") as f:
        # write metadata
        write_delimited(f, meta_data_proto)
        f.seek(INITIAL_POS)
        write_delimited(f, brain_param_proto)

        for agent in agent_info_protos:
            write_delimited(f, agent)

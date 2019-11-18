import pathlib
import shutil
import os
from mlagents.envs.brain import BrainParameters
from mlagents.envs.communicator_objects_old.agent_info_pb2 import (
    AgentInfoProto as OldAgentInfoProto,
)
from mlagents.envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents.envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore
from google.protobuf.internal.encoder import _VarintBytes


def convert_demonstration(file_path: str, output_path: str) -> None:
    """
    Loads and parses a demonstration file.
    :param file_path: Location of demonstration file (.demo).
    :return: BrainParameter and list of BrainInfos containing demonstration data.
    """
    output_path = shutil.copyfile(file_path, output_path)
    # First 32 bytes of file dedicated to meta-data.
    INITIAL_POS = 33
    file_paths = []
    if os.path.isdir(file_path):
        all_files = os.listdir(file_path)
        for _file in all_files:
            if _file.endswith(".demo"):
                file_paths.append(os.path.join(file_path, _file))
        if not all_files:
            raise ValueError("There are no '.demo' files in the provided directory.")
    elif os.path.isfile(file_path):
        file_paths.append(file_path)
        file_extension = pathlib.Path(file_path).suffix
        if file_extension != ".demo":
            raise ValueError(
                "The file is not a '.demo' file. Please provide a file with the "
                "correct extension."
            )
    else:
        raise FileNotFoundError(
            "The demonstration file or directory {} does not exist.".format(file_path)
        )

    brain_params = None
    brain_param_proto = None
    brain_infos = []
    total_expected = 0
    for _file_path in file_paths:
        data = open(_file_path, "rb").read()
        # output_data = bytearray(open(_file_path, "rb").read())
        out_file = open(output_path, "wb")
        # out_pos = None
        next_pos, pos, obs_decoded = 0, 0, 0
        while pos < len(data):
            next_pos, pos = _DecodeVarint32(data, pos)
            if obs_decoded == 0:
                meta_data_proto = DemonstrationMetaProto()
                meta_data_proto.ParseFromString(data[pos : pos + next_pos])
                total_expected += meta_data_proto.number_steps
                pos = INITIAL_POS
            if obs_decoded == 1:
                brain_param_proto = BrainParametersProto()
                brain_param_proto.ParseFromString(data[pos : pos + next_pos])

                pos += next_pos
                out_file.write(data[0:pos])
                # out_pos = pos
            if obs_decoded > 1:
                agent_info = OldAgentInfoProto()
                agent_info.ParseFromString(data[pos : pos + next_pos])
                if brain_params is None:
                    brain_params = BrainParameters.from_proto(
                        brain_param_proto, agent_info
                    )

                next_agent_info = OldAgentInfoProto()
                if pos + next_pos < len(data):
                    _delta, _next_pos = _DecodeVarint32(data, pos + next_pos)
                    next_agent_info.ParseFromString(
                        data[_next_pos : _next_pos + _delta]
                    )

                data_to_write = AgentInfoActionPairProto()
                data_to_write.agent_info.done = agent_info.done
                data_to_write.agent_info.reward = agent_info.reward
                data_to_write.agent_info.max_step_reached = agent_info.max_step_reached
                data_to_write.agent_info.id = agent_info.id
                data_to_write.agent_info.action_mask.extend(agent_info.action_mask)
                data_to_write.agent_info.observations.extend(agent_info.observations)

                data_to_write.action_info.vector_actions.extend(
                    next_agent_info.stored_vector_actions
                )

                string_to_write = data_to_write.SerializeToString()

                string_to_write_length = data_to_write.ByteSize()
                out_file.write(_VarintBytes(string_to_write_length))
                out_file.write(string_to_write)
                # output_data[out_pos : out_pos + 1] = _VarintBytes(
                #     string_to_write_length
                # )
                # out_pos += 1
                # output_data[
                #     out_pos : out_pos + string_to_write_length
                # ] = string_to_write
                # out_pos += string_to_write_length

                if len(brain_infos) == total_expected:
                    break
                pos += next_pos
            obs_decoded += 1
        # byyy = bytes(output_data)
        # open(output_path, "wb").write(byyy)
        # open(output_path, "wb").write(byt
    return

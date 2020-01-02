import pathlib
import logging
import os
from typing import List, Tuple
import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.brain import BrainParameters, BrainInfo
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import (
    DemonstrationMetaProto,
)
from mlagents_envs.timers import timed, hierarchical_timer
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore


logger = logging.getLogger("mlagents.trainers")


@timed
def make_demo_buffer(
    pair_infos: List[AgentInfoActionPairProto],
    brain_params: BrainParameters,
    sequence_length: int,
) -> AgentBuffer:
    # Create and populate buffer using experiences
    demo_raw_buffer = AgentBuffer()
    demo_processed_buffer = AgentBuffer()
    for idx, current_pair_info in enumerate(pair_infos):
        if idx > len(pair_infos) - 2:
            break
        next_pair_info = pair_infos[idx + 1]
        current_brain_info = BrainInfo.from_agent_proto(
            0, [current_pair_info.agent_info], brain_params
        )
        next_brain_info = BrainInfo.from_agent_proto(
            0, [next_pair_info.agent_info], brain_params
        )
        previous_action = (
            np.array(pair_infos[idx].action_info.vector_actions, dtype=np.float32) * 0
        )
        if idx > 0:
            previous_action = np.array(
                pair_infos[idx - 1].action_info.vector_actions, dtype=np.float32
            )
        demo_raw_buffer["done"].append(next_brain_info.local_done[0])
        demo_raw_buffer["rewards"].append(next_brain_info.rewards[0])
        for i in range(brain_params.number_visual_observations):
            demo_raw_buffer["visual_obs%d" % i].append(
                current_brain_info.visual_observations[i][0]
            )
        if brain_params.vector_observation_space_size > 0:
            demo_raw_buffer["vector_obs"].append(
                current_brain_info.vector_observations[0]
            )
        demo_raw_buffer["actions"].append(current_pair_info.action_info.vector_actions)
        demo_raw_buffer["prev_action"].append(previous_action)
        if next_brain_info.local_done[0]:
            demo_raw_buffer.resequence_and_append(
                demo_processed_buffer, batch_size=None, training_length=sequence_length
            )
            demo_raw_buffer.reset_agent()
    demo_raw_buffer.resequence_and_append(
        demo_processed_buffer, batch_size=None, training_length=sequence_length
    )
    return demo_processed_buffer


@timed
def demo_to_buffer(
    file_path: str, sequence_length: int
) -> Tuple[BrainParameters, AgentBuffer]:
    """
    Loads demonstration file and uses it to fill training buffer.
    :param file_path: Location of demonstration file (.demo).
    :param sequence_length: Length of trajectories to fill buffer.
    :return:
    """
    brain_params, info_action_pair, _ = load_demonstration(file_path)
    demo_buffer = make_demo_buffer(info_action_pair, brain_params, sequence_length)
    return brain_params, demo_buffer


@timed
def load_demonstration(
    file_path: str
) -> Tuple[BrainParameters, List[AgentInfoActionPairProto], int]:
    """
    Loads and parses a demonstration file.
    :param file_path: Location of demonstration file (.demo).
    :return: BrainParameter and list of AgentInfoActionPairProto containing demonstration data.
    """

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
                    total_expected += meta_data_proto.number_steps
                    pos = INITIAL_POS
                if obs_decoded == 1:
                    brain_param_proto = BrainParametersProto()
                    brain_param_proto.ParseFromString(data[pos : pos + next_pos])
                    pos += next_pos
                if obs_decoded > 1:
                    agent_info_action = AgentInfoActionPairProto()
                    agent_info_action.ParseFromString(data[pos : pos + next_pos])
                    if brain_params is None:
                        brain_params = BrainParameters.from_proto(
                            brain_param_proto, agent_info_action.agent_info
                        )
                    info_action_pairs.append(agent_info_action)
                    if len(info_action_pairs) == total_expected:
                        break
                    pos += next_pos
                obs_decoded += 1
    if not brain_params:
        raise RuntimeError(
            f"No BrainParameters found in demonstration file at {file_path}."
        )
    return brain_params, info_action_pairs, total_expected

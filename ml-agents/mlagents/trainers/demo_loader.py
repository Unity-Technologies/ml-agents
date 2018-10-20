import pathlib
import logging
import os
from mlagents.trainers.buffer import Buffer
from mlagents.envs.brain import BrainParameters, BrainInfo
from mlagents.envs.communicator_objects import *
from google.protobuf.internal.decoder import _DecodeVarint32

logger = logging.getLogger("mlagents.trainers")


def make_demo_buffer(brain_infos, brain_params, sequence_length):
    # Create and populate buffer using experiences
    demo_buffer = Buffer()
    for idx, experience in enumerate(brain_infos):
        if idx > len(brain_infos) - 2:
            break
        current_brain_info = brain_infos[idx]
        next_brain_info = brain_infos[idx + 1]
        demo_buffer[0].last_brain_info = current_brain_info
        demo_buffer[0]['done'].append(next_brain_info.local_done[0])
        demo_buffer[0]['rewards'].append(next_brain_info.rewards[0])
        for i in range(brain_params.number_visual_observations):
            demo_buffer[0]['visual_obs%d' % i] \
                .append(current_brain_info.visual_observations[i][0])
        if brain_params.vector_observation_space_size > 0:
            demo_buffer[0]['vector_obs'] \
                .append(current_brain_info.vector_observations[0])
        demo_buffer[0]['actions'].append(next_brain_info.previous_vector_actions[0])
        if next_brain_info.local_done[0]:
            demo_buffer.append_update_buffer(0, batch_size=None,
                                             training_length=sequence_length)
            demo_buffer.reset_local_buffers()
    demo_buffer.append_update_buffer(0, batch_size=None,
                                     training_length=sequence_length)
    return demo_buffer


def demo_to_buffer(file_path, sequence_length):
    """
    Loads demonstration file and uses it to fill training buffer.
    :param file_path: Location of demonstration file (.demo).
    :param sequence_length: Length of trajectories to fill buffer.
    :return:
    """
    brain_params, brain_infos, _ = load_demonstration(file_path)
    demo_buffer = make_demo_buffer(brain_infos, brain_params, sequence_length)
    return brain_params, demo_buffer


def load_demonstration(file_path):
    """
    Loads and parses a demonstration file.
    :param file_path: Location of demonstration file (.demo).
    :return: BrainParameter and list of BrainInfos containing demonstration data.
    """

    # First 32 bytes of file dedicated to meta-data.
    INITIAL_POS = 33

    if not os.path.isfile(file_path):
        raise FileNotFoundError("The demonstration file {} does not exist.".format(file_path))
    file_extension = pathlib.Path(file_path).suffix
    if file_extension != '.demo':
        raise ValueError("The file is not a '.demo' file. Please provide a file with the "
                         "correct extension.")

    brain_params = None
    brain_infos = []
    data = open(file_path, "rb").read()
    next_pos, pos, obs_decoded = 0, 0, 0
    total_expected = 0
    while pos < len(data):
        next_pos, pos = _DecodeVarint32(data, pos)
        if obs_decoded == 0:
            meta_data_proto = DemonstrationMetaProto()
            meta_data_proto.ParseFromString(data[pos:pos + next_pos])
            total_expected = meta_data_proto.number_steps
            pos = INITIAL_POS
        if obs_decoded == 1:
            brain_param_proto = BrainParametersProto()
            brain_param_proto.ParseFromString(data[pos:pos + next_pos])
            brain_params = BrainParameters.from_proto(brain_param_proto)
            pos += next_pos
        if obs_decoded > 1:
            agent_info = AgentInfoProto()
            agent_info.ParseFromString(data[pos:pos + next_pos])
            brain_info = BrainInfo.from_agent_proto([agent_info], brain_params)
            brain_infos.append(brain_info)
            if len(brain_infos) == total_expected:
                break
            pos += next_pos
        obs_decoded += 1
    return brain_params, brain_infos, total_expected

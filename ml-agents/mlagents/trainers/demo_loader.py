import json
import numpy as np
import pathlib
import logging
from PIL import Image
import io
from mlagents.trainers.buffer import Buffer
from mlagents.envs.brain import BrainParameters, BrainInfo
from mlagents.envs.utilities import process_pixels
from mlagents.envs.communicator_objects import *
from google.protobuf.internal.decoder import _DecodeVarint32

logger = logging.getLogger("mlagents.trainers")


def brain_param_proto_to_obj(brain_param_proto):
    resolution = [{
        "height": x.height,
        "width": x.width,
        "blackAndWhite": x.gray_scale
    } for x in brain_param_proto.camera_resolutions]
    brain_params = BrainParameters(brain_param_proto.brain_name, {
        "vectorObservationSize": brain_param_proto.vector_observation_size,
        "numStackedVectorObservations": brain_param_proto.num_stacked_vector_observations,
        "cameraResolutions": resolution,
        "vectorActionSize": brain_param_proto.vector_action_size,
        "vectorActionDescriptions": brain_param_proto.vector_action_descriptions,
        "vectorActionSpaceType": brain_param_proto.vector_action_space_type
    })
    return brain_params


def agent_info_proto_to_brain_info(agent_info, brain_params):
    vis_obs = []
    agent_info_list = [agent_info]
    for i in range(brain_params.number_visual_observations):
        obs = [process_pixels(x.visual_observations[i],
                              brain_params.camera_resolutions[i]['blackAndWhite'])
               for x in agent_info_list]
        vis_obs += [np.array(obs)]
    if len(agent_info_list) == 0:
        memory_size = 0
    else:
        memory_size = max([len(x.memories) for x in agent_info_list])
    if memory_size == 0:
        memory = np.zeros((0, 0))
    else:
        [x.memories.extend([0] * (memory_size - len(x.memories))) for x in agent_info_list]
        memory = np.array([x.memories for x in agent_info_list])
    total_num_actions = sum(brain_params.vector_action_space_size)
    mask_actions = np.ones((len(agent_info_list), total_num_actions))
    for agent_index, agent_info in enumerate(agent_info_list):
        if agent_info.action_mask is not None:
            if len(agent_info.action_mask) == total_num_actions:
                mask_actions[agent_index, :] = [
                    0 if agent_info.action_mask[k] else 1 for k in range(total_num_actions)]
    if any([np.isnan(x.reward) for x in agent_info_list]):
        logger.warning("An agent had a NaN reward.")
    if any([np.isnan(x.stacked_vector_observation).any() for x in agent_info_list]):
        logger.warning("An agent had a NaN observation.")
    brain_info = BrainInfo(
        visual_observation=vis_obs,
        vector_observation=np.nan_to_num(
            np.array([x.stacked_vector_observation for x in agent_info_list])),
        text_observations=[x.text_observation for x in agent_info_list],
        memory=memory,
        reward=[x.reward if not np.isnan(x.reward) else 0 for x in agent_info_list],
        agents=[x.id for x in agent_info_list],
        local_done=[x.done for x in agent_info_list],
        vector_action=np.array([x.stored_vector_actions for x in agent_info_list]),
        text_action=[x.stored_text_actions for x in agent_info_list],
        max_reached=[x.max_step_reached for x in agent_info_list],
        action_mask=mask_actions
    )
    return brain_info


def make_demo_buffer(brain_infos, brain_params, sequence_length):
    # Create and populate buffer using experiences
    demo_buffer = Buffer()
    for idx, experience in enumerate(brain_infos):
        if idx > len(brain_infos) - 2:
            break
        current_brain_info = brain_infos[idx]
        next_brain_info = brain_infos[idx + 1]
        demo_buffer[0].last_brain_info = current_brain_info
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
    demo_buffer.append_update_buffer(0, batch_size=None,
                                     training_length=sequence_length)
    return demo_buffer


def load_demonstration(file_path, sequence_length):
    """
    Loads and parses a demonstration file.
    :param sequence_length: Desired sequence length for buffer.
    :param file_path: Location of demonstration file (.demo).
    :return: BrainParameter and Buffer objects containing demonstration data.
    """
    INITIAL_POS = 21

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
            brain_params = brain_param_proto_to_obj(brain_param_proto)
            pos += next_pos
        if obs_decoded > 1:
            agent_info = AgentInfoProto()
            agent_info.ParseFromString(data[pos:pos + next_pos])
            brain_info = agent_info_proto_to_brain_info(agent_info, brain_params)
            brain_infos.append(brain_info)
            if len(brain_infos) == total_expected:
                break
            pos += next_pos
        obs_decoded += 1

    demo_buffer = make_demo_buffer(brain_infos, brain_params, sequence_length)
    return brain_params, demo_buffer

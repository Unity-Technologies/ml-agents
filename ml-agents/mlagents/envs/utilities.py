from PIL import Image
import numpy as np
import io
import logging

from mlagents.envs.brain import BrainParameters, BrainInfo

logger = logging.getLogger("mlagents.envs")


def process_pixels(image_bytes, grey_scale):
    """
    Converts byte array observation image into numpy array, re-sizes it,
    and optionally converts it to grey scale
    :param grey_scale: Whether to convert the image to greyscale.
    :param image_bytes: input byte array corresponding to image
    :return: processed numpy array of observation from environment
    """
    s = bytearray(image_bytes)
    image = Image.open(io.BytesIO(s))
    s = np.array(image) / 255.0
    if grey_scale:
        s = np.mean(s, axis=2)
        s = np.reshape(s, [s.shape[0], s.shape[1], 1])
    return s


def brain_param_proto_to_obj(brain_param_proto):
    """
    Converts brain parameter proto to BrainParameter object.
    :param brain_param_proto: protobuf object.
    :return: BrainParameter object.
    """
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


def agent_info_proto_to_brain_info(agent_info_list, brain_params):
    """
    Converts list of agent infos to BrainInfo.
    """
    vis_obs = []
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
        logger.warning("An agent had a NaN reward for brain " + brain_params.brain_name)
    if any([np.isnan(x.stacked_vector_observation).any() for x in agent_info_list]):
        logger.warning("An agent had a NaN observation for brain " + brain_params.brain_name)
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

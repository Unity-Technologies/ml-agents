from mlagents_envs.base_env import AgentGroupSpec, ActionType, BatchedStepResult
from mlagents_envs.timers import hierarchical_timer, timed
from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
import logging
import numpy as np
import io
from typing import cast, List, Tuple, Union, Collection
from PIL import Image

logger = logging.getLogger("mlagents_envs")


def agent_group_spec_from_proto(
    brain_param_proto: BrainParametersProto, agent_info: AgentInfoProto
) -> AgentGroupSpec:
    """
    Converts brain parameter and agent info proto to AgentGroupSpec object.
    :param brain_param_proto: protobuf object.
    :param agent_info: protobuf object.
    :return: AgentGroupSpec object.
    """
    observation_shape = [tuple(obs.shape) for obs in agent_info.observations]
    action_type = (
        ActionType.DISCRETE
        if brain_param_proto.vector_action_space_type == 0
        else ActionType.CONTINUOUS
    )
    if action_type == ActionType.CONTINUOUS:
        action_shape: Union[
            int, Tuple[int, ...]
        ] = brain_param_proto.vector_action_size[0]
    else:
        action_shape = tuple(brain_param_proto.vector_action_size)
    return AgentGroupSpec(observation_shape, action_type, action_shape)


@timed
def process_pixels(image_bytes: bytes, gray_scale: bool) -> np.ndarray:
    """
    Converts byte array observation image into numpy array, re-sizes it,
    and optionally converts it to grey scale
    :param gray_scale: Whether to convert the image to grayscale.
    :param image_bytes: input byte array corresponding to image
    :return: processed numpy array of observation from environment
    """
    with hierarchical_timer("image_decompress"):
        image_bytearray = bytearray(image_bytes)
        image = Image.open(io.BytesIO(image_bytearray))
        # Normally Image loads lazily, this forces it to do loading in the timer scope.
        image.load()
    s = np.array(image) / 255.0
    if gray_scale:
        s = np.mean(s, axis=2)
        s = np.reshape(s, [s.shape[0], s.shape[1], 1])
    return s


@timed
def _process_visual_observation(
    obs_index: int,
    shape: Tuple[int, int, int],
    agent_info_list: Collection[
        AgentInfoProto
    ],  # pylint: disable=unsubscriptable-object
) -> np.ndarray:
    if len(agent_info_list) == 0:
        return np.zeros((0, shape[0], shape[1], shape[2]), dtype=np.float32)

    gray_scale = shape[2] == 1
    batched_visual = [
        process_pixels(agent_obs.observations[obs_index].compressed_data, gray_scale)
        for agent_obs in agent_info_list
    ]
    return np.array(batched_visual, dtype=np.float32)


@timed
def _process_vector_observation(
    obs_index: int,
    shape: Tuple[int, ...],
    agent_info_list: Collection[
        AgentInfoProto
    ],  # pylint: disable=unsubscriptable-object
) -> np.ndarray:
    if len(agent_info_list) == 0:
        return np.zeros((0, shape[0]), dtype=np.float32)
    np_obs = np.array(
        [
            agent_obs.observations[obs_index].float_data.data
            for agent_obs in agent_info_list
        ],
        dtype=np.float32,
    )
    # Check for NaNs or infs in the observations
    # If there's a NaN in the observations, the np.mean() result will be NaN
    # If there's an Inf (either sign) then the result will be Inf
    # See https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy for background
    # Note that a very large values (larger than sqrt(float_max)) will result in an Inf value here
    # This is OK though, worst case it results in an unnecessary (but harmless) nan_to_num call.
    d = np.mean(np_obs)
    has_nan = np.isnan(d)
    has_inf = not np.isfinite(d)

    # In we have any NaN or Infs, use np.nan_to_num to replace these with finite values
    if has_nan or has_inf:
        np_obs = np.nan_to_num(np_obs)

    if has_nan:
        logger.warning(f"An agent had a NaN observation in the environment")
    return np_obs


@timed
def batched_step_result_from_proto(
    agent_info_list: Collection[
        AgentInfoProto
    ],  # pylint: disable=unsubscriptable-object
    group_spec: AgentGroupSpec,
) -> BatchedStepResult:
    obs_list: List[np.ndarray] = []
    for obs_index, obs_shape in enumerate(group_spec.observation_shapes):
        is_visual = len(obs_shape) == 3
        if is_visual:
            obs_shape = cast(Tuple[int, int, int], obs_shape)
            obs_list += [
                _process_visual_observation(obs_index, obs_shape, agent_info_list)
            ]
        else:
            obs_list += [
                _process_vector_observation(obs_index, obs_shape, agent_info_list)
            ]
    rewards = np.array(
        [agent_info.reward for agent_info in agent_info_list], dtype=np.float32
    )

    d = np.dot(rewards, rewards)
    has_nan = np.isnan(d)
    has_inf = not np.isfinite(d)
    # In we have any NaN or Infs, use np.nan_to_num to replace these with finite values
    if has_nan or has_inf:
        rewards = np.nan_to_num(rewards)
    if has_nan:
        logger.warning(f"An agent had a NaN reward in the environment")

    done = np.array([agent_info.done for agent_info in agent_info_list], dtype=np.bool)
    max_step = np.array(
        [agent_info.max_step_reached for agent_info in agent_info_list], dtype=np.bool
    )
    agent_id = np.array(
        [agent_info.id for agent_info in agent_info_list], dtype=np.int32
    )
    action_mask = None
    if group_spec.is_action_discrete():
        if any([agent_info.action_mask is not None] for agent_info in agent_info_list):
            n_agents = len(agent_info_list)
            a_size = np.sum(group_spec.discrete_action_branches)
            mask_matrix = np.ones((n_agents, a_size), dtype=np.bool)
            for agent_index, agent_info in enumerate(agent_info_list):
                if agent_info.action_mask is not None:
                    if len(agent_info.action_mask) == a_size:
                        mask_matrix[agent_index, :] = [
                            False if agent_info.action_mask[k] else True
                            for k in range(a_size)
                        ]
            action_mask = (1 - mask_matrix).astype(np.bool)
            indices = _generate_split_indices(group_spec.discrete_action_branches)
            action_mask = np.split(action_mask, indices, axis=1)
    return BatchedStepResult(obs_list, rewards, done, max_step, agent_id, action_mask)


def _generate_split_indices(dims):
    if len(dims) <= 1:
        return ()
    result = (dims[0],)
    for i in range(len(dims) - 2):
        result += (dims[i + 1] + result[i],)
    return result

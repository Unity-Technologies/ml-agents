from mlagents_envs.base_env import (
    BehaviorSpec,
    ActionType,
    DecisionSteps,
    TerminalSteps,
)
from mlagents_envs.exception import UnityObservationException
from mlagents_envs.timers import hierarchical_timer, timed
from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.observation_pb2 import (
    ObservationProto,
    NONE as COMPRESSION_TYPE_NONE,
)
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
import numpy as np
import io
from typing import cast, List, Tuple, Union, Collection, Optional, Iterable
from PIL import Image


PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def behavior_spec_from_proto(
    brain_param_proto: BrainParametersProto, agent_info: AgentInfoProto
) -> BehaviorSpec:
    """
    Converts brain parameter and agent info proto to BehaviorSpec object.
    :param brain_param_proto: protobuf object.
    :param agent_info: protobuf object.
    :return: BehaviorSpec object.
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
    return BehaviorSpec(observation_shape, action_type, action_shape)


@timed
def process_pixels(image_bytes: bytes, expected_channels: int) -> np.ndarray:
    """
    Converts byte array observation image into numpy array, re-sizes it,
    and optionally converts it to grey scale
    :param image_bytes: input byte array corresponding to image
    :param expected_channels: Expected output channels
    :return: processed numpy array of observation from environment
    """
    image_bytearray = bytearray(image_bytes)

    if expected_channels == 1:
        # Convert to grayscale
        with hierarchical_timer("image_decompress"):
            image = Image.open(io.BytesIO(image_bytearray))
            # Normally Image loads lazily, load() forces it to do loading in the timer scope.
            image.load()
        s = np.array(image, dtype=np.float32) / 255.0
        s = np.mean(s, axis=2)
        s = np.reshape(s, [s.shape[0], s.shape[1], 1])
        return s

    image_arrays = []

    bytes_read = 0
    # Read the images back from the bytes (without knowing the sizes).
    while True:
        # TODO avoid creating a new array here. Unfortunately, Pillow doesn't respect the current state of the buffer
        # and always starts with seek(0), but we should be able to wrap BytesIO with something that lets us adjust
        # the "start" offset.
        buffer = io.BytesIO(image_bytearray[bytes_read:])
        with hierarchical_timer("image_decompress"):
            image = Image.open(buffer)
            image.load()
        image_arrays.append(np.array(image, dtype=np.float32) / 255.0)

        # Look for the next header, starting from the current stream location
        try:
            offset = buffer.getvalue().index(PNG_HEADER, buffer.tell())
            bytes_read += offset
        except ValueError:
            # Didn't find the header, so must be at the end.
            break

    img = np.concatenate(image_arrays, axis=2)
    # We can drop additional channels since they may need to be added to include
    # numbers of observation channels not divisible by 3.
    actual_channels = list(img.shape)[2]
    if actual_channels > expected_channels:
        img = img[..., 0:expected_channels]

    return img


@timed
def observation_to_np_array(
    obs: ObservationProto, expected_shape: Optional[Iterable[int]] = None
) -> np.ndarray:
    """
    Converts observation proto into numpy array of the appropriate size.
    :param obs: observation proto to be converted
    :param expected_shape: optional shape information, used for sanity checks.
    :return: processed numpy array of observation from environment
    """
    if expected_shape is not None:
        if list(obs.shape) != list(expected_shape):
            raise UnityObservationException(
                f"Observation did not have the expected shape - got {obs.shape} but expected {expected_shape}"
            )
    expected_channels = obs.shape[2]
    if obs.compression_type == COMPRESSION_TYPE_NONE:
        img = np.array(obs.float_data.data, dtype=np.float32)
        img = np.reshape(img, obs.shape)
        return img
    else:
        img = process_pixels(obs.compressed_data, expected_channels)
        # Compare decompressed image size to observation shape and make sure they match
        if list(obs.shape) != list(img.shape):
            raise UnityObservationException(
                f"Decompressed observation did not have the expected shape - "
                f"decompressed had {img.shape} but expected {obs.shape}"
            )
        return img


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

    batched_visual = [
        observation_to_np_array(agent_obs.observations[obs_index], shape)
        for agent_obs in agent_info_list
    ]
    return np.array(batched_visual, dtype=np.float32)


def _raise_on_nan_and_inf(data: np.array, source: str) -> np.array:
    # Check for NaNs or Infinite values in the observation or reward data.
    # If there's a NaN in the observations, the np.mean() result will be NaN
    # If there's an Infinite value (either sign) then the result will be Inf
    # See https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy for background
    # Note that a very large values (larger than sqrt(float_max)) will result in an Inf value here
    # Raise a Runtime error in the case that NaNs or Infinite values make it into the data.
    if data.size == 0:
        return data

    d = np.mean(data)
    has_nan = np.isnan(d)
    has_inf = not np.isfinite(d)

    if has_nan:
        raise RuntimeError(f"The {source} provided had NaN values.")
    if has_inf:
        raise RuntimeError(f"The {source} provided had Infinite values.")


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
    _raise_on_nan_and_inf(np_obs, "observations")
    return np_obs


@timed
def steps_from_proto(
    agent_info_list: Collection[
        AgentInfoProto
    ],  # pylint: disable=unsubscriptable-object
    behavior_spec: BehaviorSpec,
) -> Tuple[DecisionSteps, TerminalSteps]:
    decision_agent_info_list = [
        agent_info for agent_info in agent_info_list if not agent_info.done
    ]
    terminal_agent_info_list = [
        agent_info for agent_info in agent_info_list if agent_info.done
    ]
    decision_obs_list: List[np.ndarray] = []
    terminal_obs_list: List[np.ndarray] = []
    for obs_index, obs_shape in enumerate(behavior_spec.observation_shapes):
        is_visual = len(obs_shape) == 3
        if is_visual:
            obs_shape = cast(Tuple[int, int, int], obs_shape)
            decision_obs_list.append(
                _process_visual_observation(
                    obs_index, obs_shape, decision_agent_info_list
                )
            )
            terminal_obs_list.append(
                _process_visual_observation(
                    obs_index, obs_shape, terminal_agent_info_list
                )
            )
        else:
            decision_obs_list.append(
                _process_vector_observation(
                    obs_index, obs_shape, decision_agent_info_list
                )
            )
            terminal_obs_list.append(
                _process_vector_observation(
                    obs_index, obs_shape, terminal_agent_info_list
                )
            )
    decision_rewards = np.array(
        [agent_info.reward for agent_info in decision_agent_info_list], dtype=np.float32
    )
    terminal_rewards = np.array(
        [agent_info.reward for agent_info in terminal_agent_info_list], dtype=np.float32
    )

    _raise_on_nan_and_inf(decision_rewards, "rewards")
    _raise_on_nan_and_inf(terminal_rewards, "rewards")

    max_step = np.array(
        [agent_info.max_step_reached for agent_info in terminal_agent_info_list],
        dtype=np.bool,
    )
    decision_agent_id = np.array(
        [agent_info.id for agent_info in decision_agent_info_list], dtype=np.int32
    )
    terminal_agent_id = np.array(
        [agent_info.id for agent_info in terminal_agent_info_list], dtype=np.int32
    )
    action_mask = None
    if behavior_spec.is_action_discrete():
        if any(
            [agent_info.action_mask is not None]
            for agent_info in decision_agent_info_list
        ):
            n_agents = len(decision_agent_info_list)
            a_size = np.sum(behavior_spec.discrete_action_branches)
            mask_matrix = np.ones((n_agents, a_size), dtype=np.bool)
            for agent_index, agent_info in enumerate(decision_agent_info_list):
                if agent_info.action_mask is not None:
                    if len(agent_info.action_mask) == a_size:
                        mask_matrix[agent_index, :] = [
                            False if agent_info.action_mask[k] else True
                            for k in range(a_size)
                        ]
            action_mask = (1 - mask_matrix).astype(np.bool)
            indices = _generate_split_indices(behavior_spec.discrete_action_branches)
            action_mask = np.split(action_mask, indices, axis=1)
    return (
        DecisionSteps(
            decision_obs_list, decision_rewards, decision_agent_id, action_mask
        ),
        TerminalSteps(terminal_obs_list, terminal_rewards, max_step, terminal_agent_id),
    )


def _generate_split_indices(dims):
    if len(dims) <= 1:
        return ()
    result = (dims[0],)
    for i in range(len(dims) - 2):
        result += (dims[i + 1] + result[i],)
    return result

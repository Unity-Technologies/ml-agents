from mlagents_envs.base_env import (
    ActionSpec,
    ObservationSpec,
    DimensionProperty,
    BehaviorSpec,
    DecisionSteps,
    TerminalSteps,
    ObservationType,
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
from typing import cast, List, Tuple, Collection, Optional, Iterable
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
    observation_specs = []
    for obs in agent_info.observations:
        observation_specs.append(
            ObservationSpec(
                name=obs.name,
                shape=tuple(obs.shape),
                observation_type=ObservationType(obs.observation_type),
                dimension_property=tuple(
                    DimensionProperty(dim) for dim in obs.dimension_properties
                )
                if len(obs.dimension_properties) > 0
                else (DimensionProperty.UNSPECIFIED,) * len(obs.shape),
            )
        )

    # proto from communicator < v1.3 does not set action spec, use deprecated fields instead
    if (
        brain_param_proto.action_spec.num_continuous_actions == 0
        and brain_param_proto.action_spec.num_discrete_actions == 0
    ):
        if brain_param_proto.vector_action_space_type_deprecated == 1:
            action_spec = ActionSpec(
                brain_param_proto.vector_action_size_deprecated[0], ()
            )
        else:
            action_spec = ActionSpec(
                0, tuple(brain_param_proto.vector_action_size_deprecated)
            )
    else:
        action_spec_proto = brain_param_proto.action_spec
        action_spec = ActionSpec(
            action_spec_proto.num_continuous_actions,
            tuple(branch for branch in action_spec_proto.discrete_branch_sizes),
        )
    return BehaviorSpec(observation_specs, action_spec)


class OffsetBytesIO:
    """
    Simple file-like class that wraps a bytes, and allows moving its "start"
    position in the bytes. This is only used for reading concatenated PNGs,
    because Pillow always calls seek(0) at the start of reading.
    """

    __slots__ = ["fp", "offset"]

    def __init__(self, data: bytes):
        self.fp = io.BytesIO(data)
        self.offset = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            res = self.fp.seek(offset + self.offset)
            return res - self.offset
        raise NotImplementedError()

    def tell(self) -> int:
        return self.fp.tell() - self.offset

    def read(self, size: int = -1) -> bytes:
        return self.fp.read(size)

    def original_tell(self) -> int:
        """
        Returns the offset into the original byte array
        """
        return self.fp.tell()


@timed
def process_pixels(
    image_bytes: bytes, expected_channels: int, mappings: Optional[List[int]] = None
) -> np.ndarray:
    """
    Converts byte array observation image into numpy array, re-sizes it,
    and optionally converts it to grey scale
    :param image_bytes: input byte array corresponding to image
    :param expected_channels: Expected output channels
    :return: processed numpy array of observation from environment
    """
    image_fp = OffsetBytesIO(image_bytes)

    image_arrays = []
    # Read the images back from the bytes (without knowing the sizes).
    while True:
        with hierarchical_timer("image_decompress"):
            image = Image.open(image_fp)
            # Normally Image loads lazily, load() forces it to do loading in the timer scope.
            image.load()
        image_arrays.append(
            np.moveaxis(np.array(image, dtype=np.float32) / 255.0, -1, 0)
        )

        # Look for the next header, starting from the current stream location
        try:
            new_offset = image_bytes.index(PNG_HEADER, image_fp.original_tell())
            image_fp.offset = new_offset
        except ValueError:
            # Didn't find the header, so must be at the end.
            break

    if mappings is not None and len(mappings) > 0:
        return _process_images_mapping(image_arrays, mappings)
    else:
        return _process_images_num_channels(image_arrays, expected_channels)


def _process_images_mapping(image_arrays, mappings):
    """
    Helper function for processing decompressed images with compressed channel mappings.
    """
    image_arrays = np.concatenate(image_arrays, axis=0).transpose((0, 1, 2))

    if len(mappings) != len(image_arrays):
        raise UnityObservationException(
            f"Compressed observation and its mapping had different number of channels - "
            f"observation had {len(image_arrays)} channels but its mapping had {len(mappings)} channels"
        )
    if len({m for m in mappings if m > -1}) != max(mappings) + 1:
        raise UnityObservationException(
            f"Invalid Compressed Channel Mapping: the mapping {mappings} does not have the correct format."
        )
    if max(mappings) >= len(image_arrays):
        raise UnityObservationException(
            f"Invalid Compressed Channel Mapping: the mapping has index larger than the total "
            f"number of channels in observation - mapping index {max(mappings)} is"
            f"invalid for input observation with {len(image_arrays)} channels."
        )

    processed_image_arrays: List[np.array] = [[] for _ in range(max(mappings) + 1)]
    for mapping_idx, img in zip(mappings, image_arrays):
        if mapping_idx > -1:
            processed_image_arrays[mapping_idx].append(img)

    for i, img_array in enumerate(processed_image_arrays):
        processed_image_arrays[i] = np.mean(img_array, axis=0)
    img = np.stack(processed_image_arrays, axis=0)
    return img


def _process_images_num_channels(image_arrays, expected_channels):
    """
    Helper function for processing decompressed images with number of expected channels.
    This is for old API without mapping provided. Use the first n channel, n=expected_channels.
    """
    if expected_channels == 1:
        # Convert to grayscale
        img = np.mean(image_arrays[0], axis=0)
        img = np.reshape(img, [1, img.shape[0], img.shape[1]])
    else:
        img = np.concatenate(image_arrays, axis=0)
        # We can drop additional channels since they may need to be added to include
        # numbers of observation channels not divisible by 3.
        actual_channels = list(img.shape)[0]
        if actual_channels > expected_channels:
            img = img[0:expected_channels, ...]
    return img


def _check_observations_match_spec(
    obs_index: int,
    observation_spec: ObservationSpec,
    agent_info_list: Collection[AgentInfoProto],
) -> None:
    """
    Check that all the observations match the expected size.
    This gives a nicer error than a cryptic numpy error later.
    """
    expected_obs_shape = tuple(observation_spec.shape)
    for agent_info in agent_info_list:
        agent_obs_shape = tuple(agent_info.observations[obs_index].shape)
        if expected_obs_shape != agent_obs_shape:
            raise UnityObservationException(
                f"Observation at index={obs_index} for agent with "
                f"id={agent_info.id} didn't match the ObservationSpec. "
                f"Expected shape {expected_obs_shape} but got {agent_obs_shape}."
            )


@timed
def _observation_to_np_array(
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
    expected_channels = obs.shape[0]
    if obs.compression_type == COMPRESSION_TYPE_NONE:
        img = np.array(obs.float_data.data, dtype=np.float32)
        img = np.reshape(img, obs.shape)
        return img
    else:
        img = process_pixels(
            obs.compressed_data, expected_channels, list(obs.compressed_channel_mapping)
        )
        # Compare decompressed image size to observation shape and make sure they match
        if list(obs.shape) != list(img.shape):
            raise UnityObservationException(
                f"Decompressed observation did not have the expected shape - "
                f"decompressed had {img.shape} but expected {obs.shape}"
            )
        return img


@timed
def _process_maybe_compressed_observation(
    obs_index: int,
    observation_spec: ObservationSpec,
    agent_info_list: Collection[AgentInfoProto],
) -> np.ndarray:
    shape = cast(Tuple[int, int, int], observation_spec.shape)
    if len(agent_info_list) == 0:
        return np.zeros((0, shape[0], shape[1], shape[2]), dtype=np.float32)

    try:
        batched_visual = [
            _observation_to_np_array(agent_obs.observations[obs_index], shape)
            for agent_obs in agent_info_list
        ]
    except ValueError:
        # Try to get a more useful error message
        _check_observations_match_spec(obs_index, observation_spec, agent_info_list)
        # If that didn't raise anything, raise the original error
        raise
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
def _process_rank_one_or_two_observation(
    obs_index: int,
    observation_spec: ObservationSpec,
    agent_info_list: Collection[AgentInfoProto],
) -> np.ndarray:
    if len(agent_info_list) == 0:
        return np.zeros((0,) + observation_spec.shape, dtype=np.float32)
    try:
        np_obs = np.array(
            [
                agent_obs.observations[obs_index].float_data.data
                for agent_obs in agent_info_list
            ],
            dtype=np.float32,
        ).reshape((len(agent_info_list),) + observation_spec.shape)
    except ValueError:
        # Try to get a more useful error message
        _check_observations_match_spec(obs_index, observation_spec, agent_info_list)
        # If that didn't raise anything, raise the original error
        raise
    _raise_on_nan_and_inf(np_obs, "observations")
    return np_obs


@timed
def steps_from_proto(
    agent_info_list: Collection[AgentInfoProto], behavior_spec: BehaviorSpec
) -> Tuple[DecisionSteps, TerminalSteps]:
    decision_agent_info_list = [
        agent_info for agent_info in agent_info_list if not agent_info.done
    ]
    terminal_agent_info_list = [
        agent_info for agent_info in agent_info_list if agent_info.done
    ]
    decision_obs_list: List[np.ndarray] = []
    terminal_obs_list: List[np.ndarray] = []
    for obs_index, observation_spec in enumerate(behavior_spec.observation_specs):
        is_visual = len(observation_spec.shape) == 3
        if is_visual:
            decision_obs_list.append(
                _process_maybe_compressed_observation(
                    obs_index, observation_spec, decision_agent_info_list
                )
            )
            terminal_obs_list.append(
                _process_maybe_compressed_observation(
                    obs_index, observation_spec, terminal_agent_info_list
                )
            )
        else:
            decision_obs_list.append(
                _process_rank_one_or_two_observation(
                    obs_index, observation_spec, decision_agent_info_list
                )
            )
            terminal_obs_list.append(
                _process_rank_one_or_two_observation(
                    obs_index, observation_spec, terminal_agent_info_list
                )
            )
    decision_rewards = np.array(
        [agent_info.reward for agent_info in decision_agent_info_list], dtype=np.float32
    )
    terminal_rewards = np.array(
        [agent_info.reward for agent_info in terminal_agent_info_list], dtype=np.float32
    )

    decision_group_rewards = np.array(
        [agent_info.group_reward for agent_info in decision_agent_info_list],
        dtype=np.float32,
    )
    terminal_group_rewards = np.array(
        [agent_info.group_reward for agent_info in terminal_agent_info_list],
        dtype=np.float32,
    )

    _raise_on_nan_and_inf(decision_rewards, "rewards")
    _raise_on_nan_and_inf(terminal_rewards, "rewards")
    _raise_on_nan_and_inf(decision_group_rewards, "group_rewards")
    _raise_on_nan_and_inf(terminal_group_rewards, "group_rewards")

    decision_group_id = [agent_info.group_id for agent_info in decision_agent_info_list]
    terminal_group_id = [agent_info.group_id for agent_info in terminal_agent_info_list]

    max_step = np.array(
        [agent_info.max_step_reached for agent_info in terminal_agent_info_list],
        dtype=bool,
    )
    decision_agent_id = np.array(
        [agent_info.id for agent_info in decision_agent_info_list], dtype=np.int32
    )
    terminal_agent_id = np.array(
        [agent_info.id for agent_info in terminal_agent_info_list], dtype=np.int32
    )
    action_mask = None
    if behavior_spec.action_spec.discrete_size > 0:
        if any(
            [agent_info.action_mask is not None]
            for agent_info in decision_agent_info_list
        ):
            n_agents = len(decision_agent_info_list)
            a_size = np.sum(behavior_spec.action_spec.discrete_branches)
            mask_matrix = np.ones((n_agents, a_size), dtype=bool)
            for agent_index, agent_info in enumerate(decision_agent_info_list):
                if agent_info.action_mask is not None:
                    if len(agent_info.action_mask) == a_size:
                        mask_matrix[agent_index, :] = [
                            False if agent_info.action_mask[k] else True
                            for k in range(a_size)
                        ]
            action_mask = (1 - mask_matrix).astype(bool)
            indices = _generate_split_indices(
                behavior_spec.action_spec.discrete_branches
            )
            action_mask = np.split(action_mask, indices, axis=1)
    return (
        DecisionSteps(
            decision_obs_list,
            decision_rewards,
            decision_agent_id,
            action_mask,
            decision_group_id,
            decision_group_rewards,
        ),
        TerminalSteps(
            terminal_obs_list,
            terminal_rewards,
            max_step,
            terminal_agent_id,
            terminal_group_id,
            terminal_group_rewards,
        ),
    )


def _generate_split_indices(dims):
    if len(dims) <= 1:
        return ()
    result = (dims[0],)
    for i in range(len(dims) - 2):
        result += (dims[i + 1] + result[i],)
    return result

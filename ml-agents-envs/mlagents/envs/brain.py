import logging
import numpy as np
import io

from mlagents.envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents.envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents.envs.timers import hierarchical_timer, timed
from typing import Dict, List, NamedTuple, Optional
from PIL import Image

logger = logging.getLogger("mlagents.envs")


class CameraResolution(NamedTuple):
    height: int
    width: int
    num_channels: int

    @property
    def gray_scale(self) -> bool:
        return self.num_channels == 1


class BrainParameters:
    def __init__(
        self,
        brain_name: str,
        vector_observation_space_size: int,
        num_stacked_vector_observations: int,
        camera_resolutions: List[CameraResolution],
        vector_action_space_size: List[int],
        vector_action_descriptions: List[str],
        vector_action_space_type: int,
    ):
        """
        Contains all brain-specific parameters.
        """
        self.brain_name = brain_name
        self.vector_observation_space_size = vector_observation_space_size
        self.num_stacked_vector_observations = num_stacked_vector_observations
        self.number_visual_observations = len(camera_resolutions)
        self.camera_resolutions = camera_resolutions
        self.vector_action_space_size = vector_action_space_size
        self.vector_action_descriptions = vector_action_descriptions
        self.vector_action_space_type = ["discrete", "continuous"][
            vector_action_space_type
        ]

    def __str__(self):
        return """Unity brain name: {}
        Number of Visual Observations (per agent): {}
        Vector Observation space size (per agent): {}
        Number of stacked Vector Observation: {}
        Vector Action space type: {}
        Vector Action space size (per agent): {}
        Vector Action descriptions: {}""".format(
            self.brain_name,
            str(self.number_visual_observations),
            str(self.vector_observation_space_size),
            str(self.num_stacked_vector_observations),
            self.vector_action_space_type,
            str(self.vector_action_space_size),
            ", ".join(self.vector_action_descriptions),
        )

    @staticmethod
    def from_proto(
        brain_param_proto: BrainParametersProto, agent_info: AgentInfoProto
    ) -> "BrainParameters":
        """
        Converts brain parameter proto to BrainParameter object.
        :param brain_param_proto: protobuf object.
        :return: BrainParameter object.
        """
        resolutions = [
            CameraResolution(x.shape[0], x.shape[1], x.shape[2])
            for x in agent_info.compressed_observations
        ]

        brain_params = BrainParameters(
            brain_param_proto.brain_name,
            brain_param_proto.vector_observation_size,
            brain_param_proto.num_stacked_vector_observations,
            resolutions,
            list(brain_param_proto.vector_action_size),
            list(brain_param_proto.vector_action_descriptions),
            brain_param_proto.vector_action_space_type,
        )
        return brain_params


class BrainInfo:
    def __init__(
        self,
        visual_observation,
        vector_observation,
        text_observations,
        memory=None,
        reward=None,
        agents=None,
        local_done=None,
        vector_action=None,
        text_action=None,
        max_reached=None,
        action_mask=None,
        custom_observations=None,
    ):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.visual_observations = visual_observation
        self.vector_observations = vector_observation
        self.text_observations = text_observations
        self.memories = memory
        self.rewards = reward
        self.local_done = local_done
        self.max_reached = max_reached
        self.agents = agents
        self.previous_vector_actions = vector_action
        self.previous_text_actions = text_action
        self.action_masks = action_mask
        self.custom_observations = custom_observations

    def merge(self, other):
        for i in range(len(self.visual_observations)):
            self.visual_observations[i].extend(other.visual_observations[i])
        self.vector_observations = np.append(
            self.vector_observations, other.vector_observations, axis=0
        )
        self.text_observations.extend(other.text_observations)
        self.memories = self.merge_memories(
            self.memories, other.memories, self.agents, other.agents
        )
        self.rewards = safe_concat_lists(self.rewards, other.rewards)
        self.local_done = safe_concat_lists(self.local_done, other.local_done)
        self.max_reached = safe_concat_lists(self.max_reached, other.max_reached)
        self.agents = safe_concat_lists(self.agents, other.agents)
        self.previous_vector_actions = safe_concat_np_ndarray(
            self.previous_vector_actions, other.previous_vector_actions
        )
        self.previous_text_actions = safe_concat_lists(
            self.previous_text_actions, other.previous_text_actions
        )
        self.action_masks = safe_concat_np_ndarray(
            self.action_masks, other.action_masks
        )
        self.custom_observations = safe_concat_lists(
            self.custom_observations, other.custom_observations
        )

    @staticmethod
    def merge_memories(m1, m2, agents1, agents2):
        if len(m1) == 0 and len(m2) != 0:
            m1 = np.zeros((len(agents1), m2.shape[1]))
        elif len(m2) == 0 and len(m1) != 0:
            m2 = np.zeros((len(agents2), m1.shape[1]))
        elif m2.shape[1] > m1.shape[1]:
            new_m1 = np.zeros((m1.shape[0], m2.shape[1]))
            new_m1[0 : m1.shape[0], 0 : m1.shape[1]] = m1
            return np.append(new_m1, m2, axis=0)
        elif m1.shape[1] > m2.shape[1]:
            new_m2 = np.zeros((m2.shape[0], m1.shape[1]))
            new_m2[0 : m2.shape[0], 0 : m2.shape[1]] = m2
            return np.append(m1, new_m2, axis=0)
        return np.append(m1, m2, axis=0)

    @staticmethod
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

    @staticmethod
    def from_agent_proto(
        worker_id: int,
        agent_info_list: List[AgentInfoProto],
        brain_params: BrainParameters,
    ) -> "BrainInfo":
        """
        Converts list of agent infos to BrainInfo.
        """
        vis_obs: List[np.ndarray] = []
        for i in range(brain_params.number_visual_observations):
            obs = [
                BrainInfo.process_pixels(
                    x.compressed_observations[i].data,
                    brain_params.camera_resolutions[i].gray_scale,
                )
                for x in agent_info_list
            ]
            vis_obs += [obs]
        if len(agent_info_list) == 0:
            memory_size = 0
        else:
            memory_size = max(len(x.memories) for x in agent_info_list)
        if memory_size == 0:
            memory = np.zeros((0, 0))
        else:
            [
                x.memories.extend([0] * (memory_size - len(x.memories)))
                for x in agent_info_list
            ]
            memory = np.array([list(x.memories) for x in agent_info_list])
        total_num_actions = sum(brain_params.vector_action_space_size)
        mask_actions = np.ones((len(agent_info_list), total_num_actions))
        for agent_index, agent_info in enumerate(agent_info_list):
            if agent_info.action_mask is not None:
                if len(agent_info.action_mask) == total_num_actions:
                    mask_actions[agent_index, :] = [
                        0 if agent_info.action_mask[k] else 1
                        for k in range(total_num_actions)
                    ]
        if any(np.isnan(x.reward) for x in agent_info_list):
            logger.warning(
                "An agent had a NaN reward for brain " + brain_params.brain_name
            )

        if len(agent_info_list) == 0:
            vector_obs = np.zeros(
                (
                    0,
                    brain_params.vector_observation_space_size
                    * brain_params.num_stacked_vector_observations,
                )
            )
        else:
            stacked_obs = []
            has_nan = False
            has_inf = False
            for x in agent_info_list:
                np_obs = np.array(x.stacked_vector_observation)
                # Check for NaNs or infs in the observations
                # If there's a NaN in the observations, the dot() result will be NaN
                # If there's an Inf (either sign) then the result will be Inf
                # See https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy for background
                # Note that a very large values (larger than sqrt(float_max)) will result in an Inf value here
                # This is OK though, worst case it results in an unnecessary (but harmless) nan_to_num call.
                d = np.dot(np_obs, np_obs)
                has_nan = has_nan or np.isnan(d)
                has_inf = has_inf or not np.isfinite(d)
                stacked_obs.append(np_obs)
            vector_obs = np.array(stacked_obs)

            # In we have any NaN or Infs, use np.nan_to_num to replace these with finite values
            if has_nan or has_inf:
                vector_obs = np.nan_to_num(vector_obs)

            if has_nan:
                logger.warning(
                    f"An agent had a NaN observation for brain {brain_params.brain_name}"
                )

        agents = [f"${worker_id}-{x.id}" for x in agent_info_list]
        brain_info = BrainInfo(
            visual_observation=vis_obs,
            vector_observation=vector_obs,
            text_observations=[x.text_observation for x in agent_info_list],
            memory=memory,
            reward=[x.reward if not np.isnan(x.reward) else 0 for x in agent_info_list],
            agents=agents,
            local_done=[x.done for x in agent_info_list],
            vector_action=np.array([x.stored_vector_actions for x in agent_info_list]),
            text_action=[list(x.stored_text_actions) for x in agent_info_list],
            max_reached=[x.max_step_reached for x in agent_info_list],
            custom_observations=[x.custom_observation for x in agent_info_list],
            action_mask=mask_actions,
        )
        return brain_info


def safe_concat_lists(l1: Optional[List], l2: Optional[List]) -> Optional[List]:
    if l1 is None:
        if l2 is None:
            return None
        else:
            return l2.copy()
    else:
        if l2 is None:
            return l1.copy()
        else:
            copy = l1.copy()
            copy.extend(l2)
            return copy


def safe_concat_np_ndarray(
    a1: Optional[np.ndarray], a2: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    if a1 is not None and a1.size != 0:
        if a2 is not None and a2.size != 0:
            return np.append(a1, a2, axis=0)
        else:
            return a1.copy()
    elif a2 is not None and a2.size != 0:
        return a2.copy()
    return None


# Renaming of dictionary of brain name to BrainInfo for clarity
AllBrainInfo = Dict[str, BrainInfo]

import logging
import numpy as np
import io

from mlagents.envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents.envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from mlagents.envs.communicator_objects.compressed_observation_pb2 import (
    ObservationProto,
)
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

    def __str__(self):
        return f"CameraResolution({self.height}, {self.width}, {self.num_channels})"


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
        Camera Resolutions: {}
        Vector Observation space size (per agent): {}
        Number of stacked Vector Observation: {}
        Vector Action space type: {}
        Vector Action space size (per agent): {}
        Vector Action descriptions: {}""".format(
            self.brain_name,
            str(self.number_visual_observations),
            str([str(cr) for cr in self.camera_resolutions]),
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
            CameraResolution(obs.shape[0], obs.shape[1], obs.shape[2])
            for obs in agent_info.observations
            if len(obs.shape) >= 3
        ]

        total_vector_obs = sum(
            obs.shape[0] for obs in agent_info.observations if len(obs.shape) == 1
        )

        assert (
            total_vector_obs
            == brain_param_proto.vector_observation_size
            * brain_param_proto.num_stacked_vector_observations
        ), f"total_vector_obs = {total_vector_obs}. brain_param_proto.vector_observation_size={brain_param_proto.vector_observation_size} brain_param_proto.num_stacked_vector_observations={brain_param_proto.num_stacked_vector_observations}"  # noqa

        brain_params = BrainParameters(
            brain_name=brain_param_proto.brain_name,
            vector_observation_space_size=brain_param_proto.vector_observation_size,
            num_stacked_vector_observations=brain_param_proto.num_stacked_vector_observations,
            camera_resolutions=resolutions,
            vector_action_space_size=list(brain_param_proto.vector_action_size),
            vector_action_descriptions=list(
                brain_param_proto.vector_action_descriptions
            ),
            vector_action_space_type=brain_param_proto.vector_action_space_type,
        )
        return brain_params


class BrainInfo:
    def __init__(
        self,
        visual_observation,
        vector_observation,
        reward=None,
        agents=None,
        local_done=None,
        vector_action=None,
        max_reached=None,
        action_mask=None,
    ):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.visual_observations = visual_observation
        self.vector_observations = vector_observation
        self.rewards = reward
        self.local_done = local_done
        self.max_reached = max_reached
        self.agents = agents
        self.previous_vector_actions = vector_action
        self.action_masks = action_mask

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
        vector_observation_protos: List[List[ObservationProto]] = []
        visual_observation_protos: List[List[ObservationProto]] = []

        # Split the observations into vector (len(shape)==1) and visual (len(shape)==3) lists
        for agent in agent_info_list:
            agent_vis: List[ObservationProto] = []
            agent_vec: List[ObservationProto] = []
            for proto_obs in agent.observations:
                is_visual = len(proto_obs.shape) == 3
                if is_visual:
                    agent_vis.append(proto_obs)
                else:
                    agent_vec.append(proto_obs)
            vector_observation_protos.append(agent_vec)
            visual_observation_protos.append(agent_vis)

        vis_obs = BrainInfo._process_visual_observations(
            brain_params, visual_observation_protos
        )

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

        vector_obs = BrainInfo._process_vector_observations(
            brain_params, agent_info_list, vector_observation_protos
        )

        agents = [f"${worker_id}-{x.id}" for x in agent_info_list]
        brain_info = BrainInfo(
            visual_observation=vis_obs,
            vector_observation=vector_obs,
            reward=[x.reward if not np.isnan(x.reward) else 0 for x in agent_info_list],
            agents=agents,
            local_done=[x.done for x in agent_info_list],
            vector_action=np.array([x.stored_vector_actions for x in agent_info_list]),
            max_reached=[x.max_step_reached for x in agent_info_list],
            action_mask=mask_actions,
        )
        return brain_info

    @staticmethod
    def _process_visual_observations(
        brain_params: BrainParameters, visual_observations: List[List[ObservationProto]]
    ) -> List[np.ndarray]:
        vis_obs: List[np.ndarray] = []
        for i in range(brain_params.number_visual_observations):
            # TODO check compression type, handle uncompressed visuals
            obs = [
                BrainInfo.process_pixels(
                    agent_obs[i].compressed_data,
                    brain_params.camera_resolutions[i].gray_scale,
                )
                for agent_obs in visual_observations
            ]
            vis_obs += [obs]
        return vis_obs

    @staticmethod
    def _process_vector_observations(
        brain_params: BrainParameters,
        agent_info_list: List[AgentInfoProto],
        vector_observations: List[List[ObservationProto]],
    ) -> np.ndarray:
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
            for x, vec_obs in zip(agent_info_list, vector_observations):
                # Concatenate vector obs
                proto_vector_obs: List[float] = []
                for vo in vec_obs:
                    # TODO consider itertools.chain here
                    proto_vector_obs.extend(vo.float_data.data)
                if proto_vector_obs:
                    # assert proto_vector_obs == x.stacked_vector_observation
                    np_obs = np.array(proto_vector_obs)
                else:
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
        return vector_obs


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

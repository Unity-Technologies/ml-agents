import logging
import numpy as np
import io

from typing import Dict, List
from PIL import Image
from mlagents.envs.communicator_objects import AgentInfoProto

logger = logging.getLogger("mlagents.envs")


class BrainParameters:
    def __init__(
        self,
        brain_name: str,
        vector_observation_space_size: int,
        num_stacked_vector_observations: int,
        camera_resolutions: List[Dict],
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
    def from_proto(brain_param_proto):
        """
        Converts brain parameter proto to BrainParameter object.
        :param brain_param_proto: protobuf object.
        :return: BrainParameter object.
        """
        resolution = [
            {"height": x.height, "width": x.width, "blackAndWhite": x.gray_scale}
            for x in brain_param_proto.camera_resolutions
        ]
        brain_params = BrainParameters(
            brain_param_proto.brain_name,
            brain_param_proto.vector_observation_size,
            brain_param_proto.num_stacked_vector_observations,
            resolution,
            list(brain_param_proto.vector_action_size),
            list(brain_param_proto.vector_action_descriptions),
            brain_param_proto.vector_action_space_type,
        )
        return brain_params


class AgentInfo:
    def __init__(
        self,
        brain_name,
        visual_observation,
        vector_observation,
        text_observation,
        memory,
        reward,
        id: str,
        local_done=None,
        previous_vector_actions=None,
        previous_text_actions=None,
        max_step_reached=None,
        action_mask=None,
        custom_observation=None,
    ):
        self.brain_name = brain_name
        self.visual_observations = visual_observation
        self.vector_observations = vector_observation
        self.text_observation = text_observation
        self.memories = memory
        self.reward = reward
        self.local_done = local_done
        self.max_step_reached = max_step_reached
        self.id = id
        self.previous_vector_actions = previous_vector_actions
        self.previous_text_actions = previous_text_actions
        self.action_mask = action_mask
        self.custom_observation = custom_observation

    @staticmethod
    def process_pixels(image_bytes: bytes, gray_scale: bool) -> np.ndarray:
        """
        Converts byte array observation image into numpy array, re-sizes it,
        and optionally converts it to grey scale
        :param gray_scale: Whether to convert the image to grayscale.
        :param image_bytes: input byte array corresponding to image
        :return: processed numpy array of observation from environment
        """
        image_bytearray = bytearray(image_bytes)
        image = Image.open(io.BytesIO(image_bytearray))
        s = np.array(image) / 255.0
        if gray_scale:
            s = np.mean(s, axis=2)
            s = np.reshape(s, [s.shape[0], s.shape[1], 1])
        return s

    @staticmethod
    def combine_memories(agent_infos: "List[AgentInfo]"):
        if len(agent_infos) == 0:
            memory_size = 0
        else:
            memory_size = max([len(x.memories) for x in agent_infos])
        if memory_size == 0:
            return np.zeros((0, 0))
        else:
            [
                x.memories.extend([0] * (memory_size - len(x.memories)))
                for x in agent_infos
            ]
            return np.array([list(x.memories) for x in agent_infos])

    @staticmethod
    def from_agent_proto(
        worker_id: int, agent_info: AgentInfoProto, brain_params: BrainParameters
    ):
        """
        Converts AgentInfoProto to an AgentInfo.
        """
        vis_obs: List[np.ndarray] = []
        for i in range(brain_params.number_visual_observations):
            obs = AgentInfo.process_pixels(
                agent_info.visual_observations[i],
                brain_params.camera_resolutions[i]["blackAndWhite"],
            )
            vis_obs += [obs]
        memory_size = len(agent_info.memories)
        if memory_size == 0:
            memory: List[float] = []
        else:
            memory = list(agent_info.memories)
        total_num_actions = sum(brain_params.vector_action_space_size)
        mask_actions = np.ones((1, total_num_actions))
        if agent_info.action_mask is not None:
            if len(agent_info.action_mask) == total_num_actions:
                mask_actions[0, :] = [
                    0 if agent_info.action_mask[k] else 1
                    for k in range(total_num_actions)
                ]
        if np.isnan(agent_info.reward):
            logger.warning(
                "An agent had a NaN reward for brain " + brain_params.brain_name
            )
        if np.isnan(agent_info.stacked_vector_observation).any():
            logger.warning(
                "An agent had a NaN observation for brain " + brain_params.brain_name
            )
        vector_obs = np.nan_to_num(
            np.array(list(agent_info.stacked_vector_observation))
        )
        agent_id = f"${worker_id}-{agent_info.id}"
        brain_info = AgentInfo(
            brain_name=brain_params.brain_name,
            visual_observation=vis_obs,
            vector_observation=vector_obs,
            text_observation=agent_info.text_observation,
            memory=memory,
            reward=agent_info.reward if not np.isnan(agent_info.reward) else 0,
            id=agent_id,
            local_done=agent_info.done,
            previous_vector_actions=np.array(list(agent_info.stored_vector_actions)),
            previous_text_actions=list(agent_info.stored_text_actions),
            max_step_reached=agent_info.max_step_reached,
            custom_observation=agent_info.custom_observation,
            action_mask=mask_actions,
        )
        return brain_info

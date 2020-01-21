import logging

from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto
from typing import List, NamedTuple

logger = logging.getLogger("mlagents.trainers")


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
        Vector Action space type: {}
        Vector Action space size (per agent): {}
        Vector Action descriptions: {}""".format(
            self.brain_name,
            str(self.number_visual_observations),
            str([str(cr) for cr in self.camera_resolutions]),
            str(self.vector_observation_space_size),
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

        brain_params = BrainParameters(
            brain_name=brain_param_proto.brain_name,
            vector_observation_space_size=total_vector_obs,
            camera_resolutions=resolutions,
            vector_action_space_size=list(brain_param_proto.vector_action_size),
            vector_action_descriptions=list(
                brain_param_proto.vector_action_descriptions
            ),
            vector_action_space_type=brain_param_proto.vector_action_space_type,
        )
        return brain_params

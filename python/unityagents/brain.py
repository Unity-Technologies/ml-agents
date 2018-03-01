from typing import Dict


class BrainInfo:
    def __init__(self, visual_observation, vector_observation, text_observations, memory=None,
                reward=None, agents=None, local_done=None,
                vector_action=None, text_action=None, max_reached=None):
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


AllBrainInfo = Dict[str, BrainInfo]


class BrainParameters:
    def __init__(self, brain_name, brain_param):
        """
        Contains all brain-specific parameters.
        :param brain_name: Name of brain.
        :param brain_param: Dictionary of brain parameters.
        """
        self.brain_name = brain_name
        self.vector_observation_space_size = brain_param["vectorObservationSize"]
        self.num_stacked_vector_observations = brain_param["numStackedVectorObservations"]
        self.number_visual_observations = len(brain_param["cameraResolutions"])
        self.camera_resolutions = brain_param["cameraResolutions"]
        self.vector_action_space_size = brain_param["vectorActionSize"]
        self.vector_action_descriptions = brain_param["vectorActionDescriptions"]
        self.vector_action_space_type = ["discrete", "continuous"][brain_param["vectorActionSpaceType"]]
        self.vector_observation_space_type = ["discrete", "continuous"][brain_param["vectorObservationSpaceType"]]

    def __str__(self):
        return '''Unity brain name: {0}
        Number of Visual Observations (per agent): {1}
        Vector Observation space type: {2}
        Vector Observation space size (per agent): {3}
        Number of stacked Vector Observation: {4}
        Vector Action space type: {5}
        Vector Action space size (per agent): {6}
        Vector Action descriptions: {7}'''.format(self.brain_name,
                                           str(self.number_visual_observations),
                                           self.vector_observation_space_type,
                                           str(self.vector_observation_space_size),
                                           str(self.num_stacked_vector_observations),
                                           self.vector_action_space_type,
                                           str(self.vector_action_space_size),
                                           ', '.join(self.vector_action_descriptions))

class BrainInfo:
    def __init__(self, observation, state, memory=None, reward=None, agents=None, local_done=None, action =None):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.observations = observation
        self.states = state
        self.memories = memory
        self.rewards = reward
        self.local_done = local_done
        self.agents = agents
        self.previous_actions = action


class BrainParameters:
    def __init__(self, brain_name, brain_param):
        """
        Contains all brain-specific parameters.
        :param brain_name: Name of brain.
        :param brain_param: Dictionary of brain parameters.
        """
        self.brain_name = brain_name
        self.state_space_size = brain_param["stateSize"]
        self.number_observations = len(brain_param["cameraResolutions"])
        self.camera_resolutions = brain_param["cameraResolutions"]
        self.action_space_size = brain_param["actionSize"]
        self.memory_space_size = brain_param["memorySize"]
        self.action_descriptions = brain_param["actionDescriptions"]
        self.action_space_type = ["discrete", "continuous"][brain_param["actionSpaceType"]]
        self.state_space_type = ["discrete", "continuous"][brain_param["stateSpaceType"]]

    def __str__(self):
        return '''Unity brain name: {0}
        Number of observations (per agent): {1}
        State space type: {2}
        State space size (per agent): {3}
        Action space type: {4}
        Action space size (per agent): {5}
        Memory space size (per agent): {6}
        Action descriptions: {7}'''.format(self.brain_name,
                                           str(self.number_observations), self.state_space_type,
                                           str(self.state_space_size), self.action_space_type,
                                           str(self.action_space_size),
                                           str(self.memory_space_size),
                                           ', '.join(self.action_descriptions))

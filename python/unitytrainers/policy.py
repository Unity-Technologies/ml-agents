import logging

from unitytrainers.trainer import UnityTrainerException

logger = logging.getLogger("unityagents")


class Policy(object):
    """
    Contains a TensorFlow model, and the necessary
    functions to interact with it to perform inference and updating
    """
    def __init__(self, seed, brain, trainer_parameters, sess):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param env: Environment.
        :param brain_name:
        :param trainer_parameters:
        :param sess:
        """
        self.m_size = None
        self.model = None
        self.inference_dict = {}
        self.update_dict = {}
        self.sequence_length = 1
        self.seed = seed
        self.brain = brain
        self.variable_scope = trainer_parameters['graph_scope']
        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.use_continuous_act = (brain.vector_action_space_type == "continuous")
        self.use_visual_obs = (brain.number_visual_observations > 0)
        self.use_vector_obs = (brain.vector_observation_space_size > 0)
        self.sess = sess
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityTrainerException("The memory size for brain {0} is 0 even "
                                            "though the trainer uses recurrent."
                                            .format(brain.brain_name))
            elif self.m_size % 4 != 0:
                raise UnityTrainerException("The memory size for brain {0} is {1} "
                                            "but it must be divisible by 4."
                                            .format(brain.brain_name, self.m_size))

    def act(self, curr_brain_info):
        """

        :param curr_brain_info:
        :return:
        """
        action = None
        return action

    def update(self, batch, n_sequences, i):
        results = None
        return results

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        return self.variable_scope

import logging

from unityagents import UnityException
from unitytrainers.models import LearningModel

logger = logging.getLogger("unityagents")


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """
    pass


class Policy(object):
    """
    Contains a learning model, and the necessary
    functions to interact with it to perform evaluate and updating.
    """

    def __init__(self, seed, brain, trainer_parameters, sess):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param brain: The corresponding Brain for this policy.
        :param trainer_parameters: The trainer parameters.
        :param sess: The current TensorFlow session.
        """
        self.m_size = None
        self.model = LearningModel(0, False, False, brain, scope='Model', seed=0)
        self.inference_dict = {}
        self.update_dict = {}
        self.sequence_length = 1
        self.seed = seed
        self.brain = brain
        self.variable_scope = trainer_parameters['graph_scope']
        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.use_continuous_act = (brain.vector_action_space_type == "continuous")
        self.sess = sess
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityPolicyException("The memory size for brain {0} is 0 even "
                                           "though the trainer uses recurrent."
                                           .format(brain.brain_name))
            elif self.m_size % 4 != 0:
                raise UnityPolicyException("The memory size for brain {0} is {1} "
                                           "but it must be divisible by 4."
                                           .format(brain.brain_name, self.m_size))

    def evaluate(self, brain_info):
        """
        Evaluates policy.
        :param brain_info: BrainInfo input to network.
        :return: Output from policy based on self.inference_dict.
        """
        raise UnityPolicyException("The evaluate function was not implemented.")

    def update(self, mini_batch, num_sequences):
        """
        Performs update of the policy.
        :param num_sequences: Number of experience trajectories in batch.
        :param mini_batch: Batch of experiences.
        :return: Results of update.
        """
        raise UnityPolicyException("The update function was not implemented.")

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        return self.variable_scope

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        step = self.sess.run(self.model.global_step)
        return step

    def increment_step(self):
        """
        Increments model step.
        """
        self.sess.run(self.model.increment_step)

    def get_inference_vars(self):
        """
        :return:list of inference var names
        """
        return list(self.inference_dict.keys())

    def get_update_vars(self):
        """
        :return:list of update var names
        """
        return list(self.update_dict.keys())

    @property
    def vis_obs_size(self):
        return self.model.vis_obs_size

    @property
    def vec_obs_size(self):
        return self.model.vec_obs_size

    @property
    def use_vis_obs(self):
        return self.model.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.model.vec_obs_size > 0

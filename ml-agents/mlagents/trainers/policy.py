import logging
import numpy as np

from mlagents.trainers import UnityException
from mlagents.trainers.models import LearningModel

logger = logging.getLogger("mlagents.trainers")


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
        self.model = None
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
        Evaluates policy for the agent experiences provided.
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

    def _execute_model(self, feed_dict, out_dict):
        """
        Executes model.
        :param feed_dict: Input dictionary mapping nodes to input data.
        :param out_dict: Output dictionary mapping names to nodes.
        :return: Dictionary mapping names to input data.
        """
        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

    def _fill_eval_dict(self, feed_dict, brain_info):
        for i, _ in enumerate(brain_info.visual_observations):
            feed_dict[self.model.visual_in[i]] = brain_info.visual_observations[i]
        if self.use_vec_obs:
            feed_dict[self.model.vector_in] = brain_info.vector_observations
        if not self.use_continuous_act:
            feed_dict[self.model.action_masks] = brain_info.action_masks
        return feed_dict

    def make_empty_memory(self, num_agents):
        """
        Creates empty memory for use with RNNs
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.m_size))

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

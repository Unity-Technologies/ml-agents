import logging

import numpy as np
import tensorflow as tf
from unitytrainers.ppo.models import PPOModel
from unitytrainers.trainer import UnityTrainerException

logger = logging.getLogger("unityagents")


class PPOPolicy(object):
    def __init__(self, seed, env, brain_name, trainer_parameters, sess):
        self.m_size = None
        self.has_updated = False
        self.variable_scope = trainer_parameters['graph_scope']
        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.is_continuous_action = (env.brains[brain_name].vector_action_space_type == "continuous")
        self.use_visual_obs = (env.brains[brain_name].number_visual_observations > 0)
        self.use_vector_obs = (env.brains[brain_name].vector_observation_space_size > 0)
        self.sess = sess
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityTrainerException("The memory size for brain {0} is 0 even though the trainer uses recurrent."
                                            .format(brain_name))
            elif self.m_size % 4 != 0:
                raise UnityTrainerException("The memory size for brain {0} is {1} but it must be divisible by 4."
                                            .format(brain_name, self.m_size))

        with tf.variable_scope(self.variable_scope):
            tf.set_random_seed(seed)
            self.model = PPOModel(env.brains[brain_name],
                                  lr=float(trainer_parameters['learning_rate']),
                                  h_size=int(trainer_parameters['hidden_units']),
                                  epsilon=float(trainer_parameters['epsilon']),
                                  beta=float(trainer_parameters['beta']),
                                  max_step=float(trainer_parameters['max_steps']),
                                  normalize=trainer_parameters['normalize'],
                                  use_recurrent=trainer_parameters['use_recurrent'],
                                  num_layers=int(trainer_parameters['num_layers']),
                                  m_size=self.m_size,
                                  use_curiosity=bool(trainer_parameters['use_curiosity']),
                                  curiosity_strength=float(trainer_parameters['curiosity_strength']),
                                  curiosity_enc_size=float(trainer_parameters['curiosity_enc_size']))

        self.inference_run_list = [self.model.output, self.model.all_log_probs, self.model.value,
                                   self.model.entropy, self.model.learning_rate]
        if self.is_continuous_action:
            self.inference_run_list.append(self.model.output_pre)
        if self.use_recurrent:
            self.inference_run_list.extend([self.model.memory_out])
        if self.is_training and self.use_vector_obs and self.trainer_parameters['normalize']:
            self.inference_run_list.extend([self.model.update_mean, self.model.update_variance])


    def act(self, observation):
        action = None
        return action

    def update(self, experiences):
        print("Done")

    def get_intrinsic_reward(self, observation):
        intrinsic_reward = None
        return intrinsic_reward

    def get_value_estimate(self, observation):
        value_estimate = None
        return value_estimate

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        return self.variable_scope

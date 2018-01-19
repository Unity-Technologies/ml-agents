import logging
import os

import numpy as np
import tensorflow as tf

from trainers.ppo_models import *
from trainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("unityagents")


# This works only with PPO
class GhostTrainer(Trainer):
    """Keeps copies of a PPOTrainer past graphs and uses them to other Trainers."""

    def __init__(self, sess, env, brain_name, trainer_parameters, training):
        """
        Responsible for saving and reusing past models.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.param_keys = ['brain_to_copy', 'is_ghost', 'new_model_freq', 'max_num_models']
        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the PPO trainer of "
                                            "brain {1}.".format(k, brain_name))

        super(GhostTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.brain_to_copy = trainer_parameters['brain_to_copy']
        self.variable_scope = trainer_parameters['graph_scope']
        self.original_brain_parameters = trainer_parameters['original_brain_parameters']
        self.new_model_freq = trainer_parameters['new_model_freq']
        self.steps = 0
        self.models = []
        self.max_num_models = trainer_parameters['max_num_models']
        self.last_model_replaced = 0
        for i in range(self.max_num_models):
            with tf.variable_scope(self.variable_scope + '_' + str(i)):
                self.models += [create_agent_model(env.brains[self.brain_to_copy],
                                                   lr=float(self.original_brain_parameters['learning_rate']),
                                                   h_size=int(self.original_brain_parameters['hidden_units']),
                                                   epsilon=float(self.original_brain_parameters['epsilon']),
                                                   beta=float(self.original_brain_parameters['beta']),
                                                   max_step=float(self.original_brain_parameters['max_steps']),
                                                   normalize=self.original_brain_parameters['normalize'],
                                                   use_recurrent=self.original_brain_parameters['use_recurrent'],
                                                   num_layers=int(self.original_brain_parameters['num_layers']),
                                                   m_size=self.original_brain_parameters)]
        self.model = self.models[0]

        self.is_continuous = (env.brains[brain_name].action_space_type == "continuous")
        self.use_observations = (env.brains[brain_name].number_observations > 0)
        self.use_states = (env.brains[brain_name].state_space_size > 0)
        self.use_recurrent = self.original_brain_parameters["use_recurrent"]
        self.summary_path = trainer_parameters['summary_path']

    def __str__(self):
        return '''Hypermarameters for the Ghost Trainer of brain {0}: \n{1}'''.format(
            self.brain_name, '\n'.join(['\t{0}:\t{1}'.format(x, self.trainer_parameters[x]) for x in self.param_keys]))

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def graph_scope(self):
        """
        Returns the graph scope of the trainer.
        """
        return None

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return 1

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return 0

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return 0

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps += 1

    def update_last_reward(self):
        """
        Updates the last reward
        """
        return

    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def take_action(self, info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :return: a tupple containing action, memories, values and an object
        to be passed to add experiences
        """

        epsi = None
        info = info[self.brain_name]
        feed_dict = {self.model.batch_size: len(info.states), self.model.sequence_length: 1}
        run_list = [self.model.output]
        if self.is_continuous:
            epsi = np.random.randn(len(info.states), self.brain.action_space_size)
            feed_dict[self.model.epsilon] = epsi
        if self.use_observations:
            for i, _ in enumerate(info.observations):
                feed_dict[self.model.observation_in[i]] = info.observations[i]
        if self.use_states:
            feed_dict[self.model.state_in] = info.states
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = info.memories
            run_list += [self.model.memory_out]
        if self.use_recurrent:
            actions, memories = self.sess.run(run_list, feed_dict=feed_dict)
        else:
            actions = self.sess.run(run_list, feed_dict=feed_dict)
            memories = None
        return (actions, memories, None, None)

    def add_experiences(self, info, next_info, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        return

    def process_experiences(self, info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        """
        return

    def end_episode(self):
        """
        A signal that the Episode has ended. We must use another version of the graph.
        """
        self.model = self.models[np.random.randint(0, self.max_num_models)]

    def is_ready_update(self):
        """
        Returns wether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        return self.steps % self.new_model_freq == 0

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        self.last_model_replaced = (self.last_model_replaced + 1) % self.max_num_models
        self.sess.run(self.update_target_graph(
            self.original_brain_parameters['graph_scope'],
            self.variable_scope + '_' + str(self.last_model_replaced))
        )
        return

    def write_summary(self, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param lesson_number: The lesson the trainer is at.
        """
        return

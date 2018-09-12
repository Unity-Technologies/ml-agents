# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Imitation)
# Contains an implementation of Behavioral Cloning Algorithm

import logging
import os

import numpy as np
import tensorflow as tf

from unityagents import AllBrainInfo
from unitytrainers.bc.models import BehavioralCloningModel
from unitytrainers.buffer import Buffer
from unitytrainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("unityagents")


class BehavioralCloningTrainer(Trainer):
    """The ImitationTrainer is an implementation of the imitation learning."""

    def __init__(self, sess, env, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.param_keys = ['brain_to_imitate', 'batch_size', 'time_horizon', 'graph_scope',
                           'summary_freq', 'max_steps', 'batches_per_epoch', 'use_recurrent', 'hidden_units',
                           'num_layers', 'sequence_length', 'memory_size']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the Imitation trainer of "
                                            "brain {1}.".format(k, brain_name))

        super(BehavioralCloningTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.variable_scope = trainer_parameters['graph_scope']
        self.brain_to_imitate = trainer_parameters['brain_to_imitate']
        self.batches_per_epoch = trainer_parameters['batches_per_epoch']
        self.use_recurrent = trainer_parameters['use_recurrent']
        self.sequence_length = 1
        self.m_size = None
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
        self.n_sequences = max(int(trainer_parameters['batch_size'] / self.sequence_length), 1)
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.stats = {'losses': [], 'episode_length': [], 'cumulative_reward': []}

        self.training_buffer = Buffer()
        self.is_continuous_action = (env.brains[brain_name].vector_action_space_type == "continuous")
        self.is_continuous_observation = (env.brains[brain_name].vector_observation_space_type == "continuous")
        self.use_visual_observations = (env.brains[brain_name].number_visual_observations > 0)
        if self.use_visual_observations:
            logger.info('Cannot use observations with imitation learning')
        self.use_vector_observations = (env.brains[brain_name].vector_observation_space_size > 0)
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)
        with tf.variable_scope(self.variable_scope):
            tf.set_random_seed(seed)
            self.model = BehavioralCloningModel(
                h_size=int(trainer_parameters['hidden_units']),
                lr=float(trainer_parameters['learning_rate']),
                n_layers=int(trainer_parameters['num_layers']),
                m_size=self.m_size,
                normalize=False,
                use_recurrent=trainer_parameters['use_recurrent'],
                brain=self.brain)
        self.inference_run_list = [self.model.sample_action]
        if self.use_recurrent:
            self.inference_run_list += [self.model.memory_out]

    def __str__(self):

        return '''Hyperparameters for the Imitation Trainer of brain {0}: \n{1}'''.format(
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
        return self.variable_scope

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return float(self.trainer_parameters['max_steps'])

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.sess.run(self.model.global_step)

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        if len(self.stats['cumulative_reward']) > 0:
            return np.mean(self.stats['cumulative_reward'])
        else:
            return 0

    def increment_step_and_update_last_reward(self):
        """
        Increment the step count of the trainer and Updates the last reward
        """
        self.sess.run(self.model.increment_step)
        return

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param all_brain_info: AllBrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(all_brain_info[self.brain_name].agents) == 0:
            return [], [], [], None

        agent_brain = all_brain_info[self.brain_name]
        feed_dict = {self.model.dropout_rate: 1.0, self.model.sequence_length: 1}

        if self.use_visual_observations:
            for i, _ in enumerate(agent_brain.visual_observations):
                feed_dict[self.model.visual_in[i]] = agent_brain.visual_observations[i]
        if self.use_vector_observations:
            feed_dict[self.model.vector_in] = agent_brain.vector_observations
        if self.use_recurrent:
            if agent_brain.memories.shape[1] == 0:
                agent_brain.memories = np.zeros((len(agent_brain.agents), self.m_size))
            feed_dict[self.model.memory_in] = agent_brain.memories
            agent_action, memories = self.sess.run(self.inference_run_list, feed_dict)
            return agent_action, memories, None, None
        else:
            agent_action = self.sess.run(self.inference_run_list, feed_dict)
        return agent_action, None, None, None

    def add_experiences(self, curr_info: AllBrainInfo, next_info: AllBrainInfo, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_info: Current AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param next_info: Next AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param take_action_outputs: The outputs of the take action method.
        """

        # Used to collect teacher experience into training buffer
        info_teacher = curr_info[self.brain_to_imitate]
        next_info_teacher = next_info[self.brain_to_imitate]
        for agent_id in info_teacher.agents:
            self.training_buffer[agent_id].last_brain_info = info_teacher

        for agent_id in next_info_teacher.agents:
            stored_info_teacher = self.training_buffer[agent_id].last_brain_info
            if stored_info_teacher is None:
                continue
            else:
                idx = stored_info_teacher.agents.index(agent_id)
                next_idx = next_info_teacher.agents.index(agent_id)
                if stored_info_teacher.text_observations[idx] != "":
                    info_teacher_record, info_teacher_reset = \
                        stored_info_teacher.text_observations[idx].lower().split(",")
                    next_info_teacher_record, next_info_teacher_reset = next_info_teacher.text_observations[idx].\
                        lower().split(",")
                    if next_info_teacher_reset == "true":
                        self.training_buffer.reset_update_buffer()
                else:
                    info_teacher_record, next_info_teacher_record = "true", "true"
                if info_teacher_record == "true" and next_info_teacher_record == "true":
                    if not stored_info_teacher.local_done[idx]:
                        if self.use_visual_observations:
                            for i, _ in enumerate(stored_info_teacher.visual_observations):
                                self.training_buffer[agent_id]['visual_observations%d' % i]\
                                    .append(stored_info_teacher.visual_observations[i][idx])
                        if self.use_vector_observations:
                            self.training_buffer[agent_id]['vector_observations']\
                                .append(stored_info_teacher.vector_observations[idx])
                        if self.use_recurrent:
                            if stored_info_teacher.memories.shape[1] == 0:
                                stored_info_teacher.memories = np.zeros((len(stored_info_teacher.agents), self.m_size))
                            self.training_buffer[agent_id]['memory'].append(stored_info_teacher.memories[idx])
                        self.training_buffer[agent_id]['actions'].append(next_info_teacher.
                                                                         previous_vector_actions[next_idx])
        info_student = curr_info[self.brain_name]
        next_info_student = next_info[self.brain_name]
        for agent_id in info_student.agents:
            self.training_buffer[agent_id].last_brain_info = info_student

        # Used to collect information about student performance.
        for agent_id in next_info_student.agents:
            stored_info_student = self.training_buffer[agent_id].last_brain_info
            if stored_info_student is None:
                continue
            else:
                next_idx = next_info_student.agents.index(agent_id)
                if agent_id not in self.cumulative_rewards:
                    self.cumulative_rewards[agent_id] = 0
                self.cumulative_rewards[agent_id] += next_info_student.rewards[next_idx]
                if not next_info_student.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1

    def process_experiences(self, current_info: AllBrainInfo, next_info: AllBrainInfo):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current AllBrainInfo
        :param next_info: Next AllBrainInfo
        """
        info_teacher = next_info[self.brain_to_imitate]
        for l in range(len(info_teacher.agents)):
            if ((info_teacher.local_done[l] or
                 len(self.training_buffer[info_teacher.agents[l]]['actions']) > self.trainer_parameters[
                 'time_horizon'])
                    and len(self.training_buffer[info_teacher.agents[l]]['actions']) > 0):
                agent_id = info_teacher.agents[l]
                self.training_buffer.append_update_buffer(agent_id, batch_size=None,
                                                          training_length=self.sequence_length)
                self.training_buffer[agent_id].reset_agent()

        info_student = next_info[self.brain_name]
        for l in range(len(info_student.agents)):
            if info_student.local_done[l]:
                agent_id = info_student.agents[l]
                self.stats['cumulative_reward'].append(
                    self.cumulative_rewards.get(agent_id, 0))
                self.stats['episode_length'].append(
                    self.episode_steps.get(agent_id, 0))
                self.cumulative_rewards[agent_id] = 0
                self.episode_steps[agent_id] = 0

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset. 
        Get only called when the academy resets.
        """
        self.training_buffer.reset_all()
        for agent_id in self.cumulative_rewards:
            self.cumulative_rewards[agent_id] = 0
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        return len(self.training_buffer.update_buffer['actions']) > self.n_sequences

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        self.training_buffer.update_buffer.shuffle()
        batch_losses = []
        for j in range(
                min(len(self.training_buffer.update_buffer['actions']) // self.n_sequences, self.batches_per_epoch)):
            _buffer = self.training_buffer.update_buffer
            start = j * self.n_sequences
            end = (j + 1) * self.n_sequences

            feed_dict = {self.model.dropout_rate: 0.5,
                         self.model.batch_size: self.n_sequences,
                         self.model.sequence_length: self.sequence_length}
            if self.is_continuous_action:
                feed_dict[self.model.true_action] = np.array(_buffer['actions'][start:end]).\
                    reshape([-1, self.brain.vector_action_space_size])
            else:
                feed_dict[self.model.true_action] = np.array(_buffer['actions'][start:end]).reshape([-1])
            if self.use_vector_observations:
                if not self.is_continuous_observation:
                    feed_dict[self.model.vector_in] = np.array(_buffer['vector_observations'][start:end])\
                        .reshape([-1, self.brain.num_stacked_vector_observations])
                else:
                    feed_dict[self.model.vector_in] = np.array(_buffer['vector_observations'][start:end])\
                        .reshape([-1, self.brain.vector_observation_space_size * self.brain.num_stacked_vector_observations])
            if self.use_visual_observations:
                for i, _ in enumerate(self.model.visual_in):
                    _obs = np.array(_buffer['visual_observations%d' % i][start:end])
                    feed_dict[self.model.visual_in[i]] = _obs
            if self.use_recurrent:
                feed_dict[self.model.memory_in] = np.zeros([self.n_sequences, self.m_size])
            loss, _ = self.sess.run([self.model.loss, self.model.update], feed_dict=feed_dict)
            batch_losses.append(loss)
        if len(batch_losses) > 0:
            self.stats['losses'].append(np.mean(batch_losses))
        else:
            self.stats['losses'].append(0)

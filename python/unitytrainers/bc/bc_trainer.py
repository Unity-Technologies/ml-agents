# # Unity ML Agents
# ## ML-Agent Learning (Imitation)
# Contains an implementation of Behavioral Cloning Algorithm

import logging
import os

import numpy as np
import tensorflow as tf

from unitytrainers.bc.bc_models import BehavioralCloningModel
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
                           'num_layers', 'sequence_length']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the Imitation trainer of "
                                            "brain {1}.".format(k, brain_name))

        super(BehavioralCloningTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.variable_scope = trainer_parameters['graph_scope']
        self.brain_to_imitate = trainer_parameters['brain_to_imitate']
        self.batch_size = trainer_parameters['batch_size']
        self.batches_per_epoch = trainer_parameters['batches_per_epoch']
        self.use_recurrent = trainer_parameters['use_recurrent']
        self.step = 0
        self.sequence_length = 1
        self.m_size = None
        if self.use_recurrent:
            self.m_size = env.brains[brain_name].memory_space_size
            self.sequence_length = trainer_parameters["sequence_length"]
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.stats = {'losses': [], 'episode_length': [], 'cumulative_reward': []}

        self.training_buffer = Buffer()
        self.is_continuous = (env.brains[brain_name].action_space_type == "continuous")
        self.use_observations = (env.brains[brain_name].number_observations > 0)
        if self.use_observations:
            logger.info('Cannot use observations with imitation learning')
        self.use_states = (env.brains[brain_name].state_space_size > 0)
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
                m_size=self.brain.memory_space_size,
                normalize=False,
                use_recurrent=trainer_parameters['use_recurrent'],
                brain=self.brain)

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
        return self.step

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

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.step += 1

    def update_last_reward(self):
        """
        Updates the last reward
        """
        return

    def take_action(self, all_brain_info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :return: a tupple containing action, memories, values and an object
        to be passed to add experiences
        """
        agent_brain = all_brain_info[self.brain_name]
        feed_dict = {self.model.dropout_rate: 1.0, self.model.sequence_length: 1}
        run_list = [self.model.sample_action]
        if self.use_observations:
            for i, _ in enumerate(agent_brain.observations):
                feed_dict[self.model.observation_in[i]] = agent_brain.observations[i]
        if self.use_states:
            feed_dict[self.model.state_in] = agent_brain.states
        if self.use_recurrent:
            feed_dict[self.model.memory_in] = agent_brain.memories
            run_list += [self.model.memory_out]
        if self.use_recurrent:
            agent_action, memories = self.sess.run(run_list, feed_dict)
            return agent_action, memories, None, None
        else:
            agent_action = self.sess.run(run_list, feed_dict)
        return agent_action, None, None, None

    def add_experiences(self, info, next_info, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        info_expert = info[self.brain_to_imitate]
        next_info_expert = next_info[self.brain_to_imitate]
        for agent_id in info_expert.agents:
            if agent_id in next_info_expert.agents:
                idx = info_expert.agents.index(agent_id)
                next_idx = next_info_expert.agents.index(agent_id)
                if not info_expert.local_done[idx]:
                    if self.use_observations:
                        for i, _ in enumerate(info.observations):
                            self.training_buffer[agent_id]['observations%d' % i].append(info_expert.observations[i][idx])
                    if self.use_states:
                        self.training_buffer[agent_id]['states'].append(info_expert.states[idx])
                    if self.use_recurrent:
                        self.training_buffer[agent_id]['memory'].append(info_expert.memories[idx])
                    self.training_buffer[agent_id]['actions'].append(next_info_expert.previous_actions[next_idx])

        info_student = next_info[self.brain_name]
        next_info_student = next_info[self.brain_name]
        for agent_id in info_student.agents:
            idx = info_student.agents.index(agent_id)
            next_idx = next_info_student.agents.index(agent_id)
            if not info_student.local_done[idx]:
                if agent_id not in self.cumulative_rewards:
                    self.cumulative_rewards[agent_id] = 0
                self.cumulative_rewards[agent_id] += next_info_student.rewards[next_idx]
                if agent_id not in self.episode_steps:
                    self.episode_steps[agent_id] = 0
                self.episode_steps[agent_id] += 1

    def process_experiences(self, info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        """
        info_expert = info[self.brain_to_imitate]
        for l in range(len(info_expert.agents)):
            if ((info_expert.local_done[l] or
                 len(self.training_buffer[info_expert.agents[l]]['actions']) > self.trainer_parameters[
                 'time_horizon'])
                    and len(self.training_buffer[info_expert.agents[l]]['actions']) > 0):
                agent_id = info_expert.agents[l]
                self.training_buffer.append_update_buffer(agent_id, batch_size=None, training_length=self.sequence_length)
                self.training_buffer[agent_id].reset_agent()

        info_student = info[self.brain_name]
        for l in range(len(info_student.agents)):
            if info_student.local_done[l]:
                agent_id = info_student.agents[l]
                self.stats['cumulative_reward'].append(self.cumulative_rewards[agent_id])
                self.stats['episode_length'].append(self.episode_steps[agent_id])
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
        return len(self.training_buffer.update_buffer['actions']) > self.batch_size

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        batch_size = self.trainer_parameters['batch_size']

        self.training_buffer.update_buffer.shuffle()
        batch_losses = []
        for j in range(
                min(len(self.training_buffer.update_buffer['actions']) // self.batch_size, self.batches_per_epoch)):
            _buffer = self.training_buffer.update_buffer
            start = j * batch_size
            end = (j + 1) * batch_size
            batch_states = np.array(_buffer['states'][start:end])
            batch_actions = np.array(_buffer['actions'][start:end])
            feed_dict = {self.model.true_action: batch_actions.reshape([-1, self.brain.action_space_size]),
                         self.model.dropout_rate: 0.5,
                         self.model.batch_size: batch_size,
                         self.model.sequence_length: self.sequence_length}
            if not self.is_continuous:
                feed_dict[self.model.state_in] = batch_states.reshape([-1, 1])
            else:
                feed_dict[self.model.state_in] = batch_states.reshape([-1, self.brain.state_space_size *
                                                                       self.brain.stacked_states])
            if self.use_observations:
                for i, _ in enumerate(self.model.observation_in):
                    _obs = np.array(_buffer['observations%d' % i][start:end])
                    (_batch, _seq, _w, _h, _c) = _obs.shape
                    feed_dict[self.model.observation_in[i]] = _obs.reshape([-1, _w, _h, _c])
            if self.use_recurrent:
                feed_dict[self.model.memory_in] = np.zeros([batch_size, self.m_size])

            loss, _ = self.sess.run([self.model.loss, self.model.update], feed_dict=feed_dict)
            batch_losses.append(loss)
        if len(batch_losses) > 0:
            self.stats['losses'].append(np.mean(batch_losses))
        else:
            self.stats['losses'].append(0)

    def write_summary(self, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param lesson_number: The lesson the trainer is at.
        """
        if (self.get_step % self.trainer_parameters['summary_freq'] == 0 and self.get_step != 0 and
                self.is_training and self.get_step <= self.get_max_steps):
            steps = self.get_step
            if len(self.stats['cumulative_reward']) > 0:
                mean_reward = np.mean(self.stats['cumulative_reward'])
                logger.info("{0} : Step: {1}. Mean Reward: {2}. Std of Reward: {3}."
                            .format(self.brain_name, steps, mean_reward, np.std(self.stats['cumulative_reward'])))
            summary = tf.Summary()
            for key in self.stats:
                if len(self.stats[key]) > 0:
                    stat_mean = float(np.mean(self.stats[key]))
                    summary.value.add(tag='Info/{}'.format(key), simple_value=stat_mean)
                    self.stats[key] = []
            summary.value.add(tag='Info/Lesson', simple_value=lesson_number)
            self.summary_writer.add_summary(summary, steps)
            self.summary_writer.flush()

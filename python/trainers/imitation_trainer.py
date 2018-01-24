# # Unity ML Agents
# ## ML-Agent Learning (Imitation)
# Contains an implementation of Imitation Learning

import logging
import os

import numpy as np
import tensorflow as tf

from trainers.buffer import Buffer
from trainers.ppo_models import *
from trainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("unityagents")

class ImitationNN(object):
    def __init__(self, state_size, action_size, h_size, lr, action_type, n_layers):
        self.state = tf.placeholder(shape=[None, state_size], dtype=tf.float32, name="state")
        hidden = tf.layers.dense(self.state, h_size, activation=tf.nn.elu)
        for i in range(n_layers):
            hidden = tf.layers.dense(hidden, h_size, activation=tf.nn.elu)
        hidden_drop = tf.layers.dropout(hidden, 0.5)
        self.output = tf.layers.dense(hidden_drop, action_size, activation=None)

        if action_type == "discrete":
            self.action_probs = tf.nn.softmax(self.output)
            self.sample_action = tf.multinomial(self.output, 1, name="action")
            self.true_action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action_oh = tf.one_hot(self.true_action, action_size)
            self.loss = tf.reduce_sum(-tf.log(self.action_probs + 1e-10) * self.action_oh)

            self.action_percent = tf.reduce_mean(tf.cast(
                tf.equal(tf.cast(tf.argmax(self.action_probs, axis=1), tf.int32), self.sample_action), tf.float32))
        else:
            self.sample_action = tf.identity(self.output, name="action")
            self.true_action = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.squared_difference(self.true_action, self.sample_action))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)



class ImitationTrainer(Trainer):
    """The ImitationTrainer is an implementation of the imitation learning."""
    def __init__(self, sess, env, brain_name, trainer_parameters, training):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.param_keys = [ 'is_imitation', 'brain_to_imitate', 'batch_size', 'time_horizon', 'graph_scope', 
            'summary_freq', 'max_steps', 'batches_per_epoch']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the Imitation trainer of "
                    "brain {1}.".format(k, brain_name))

        super(ImitationTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.variable_scope = trainer_parameters['graph_scope']
        self.brain_to_imitate = trainer_parameters['brain_to_imitate']
        self.batch_size = trainer_parameters['batch_size']
        self.batches_per_epoch = trainer_parameters['batches_per_epoch']
        self.step = 0
        self.cumulative_rewards = {}
        self.episode_steps = {}
        

        self.stats = {'losses': [], 'episode_length': [], 'cumulative_reward' : []}

        self.training_buffer = Buffer()
        self.is_continuous = (env.brains[brain_name].action_space_type == "continuous")
        self.use_observations = (env.brains[brain_name].number_observations > 0)
        if self.use_observations:
            logger.log('Cannot use observations with imitation learning')
        self.use_states = (env.brains[brain_name].state_space_size > 0)
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)
        s_size = self.brain.state_space_size * self.brain.stacked_states
        a_size = self.brain.action_space_size
        with tf.variable_scope(self.variable_scope):
            self.network = ImitationNN(state_size = s_size,
                     action_size = a_size, 
                     h_size = int(trainer_parameters['hidden_units']), 
                     lr = float(trainer_parameters['learning_rate']), 
                     action_type = self.brain.action_space_type, 
                     n_layers=int(trainer_parameters['num_layers']))


    def __str__(self):

        return '''Hypermarameters for the Imitation Trainer of brain {0}: \n{1}'''.format(
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


    def take_action(self, info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :return: a tupple containing action, memories, values and an object
        to be passed to add experiences
        """
        E = info[self.brain_name]
        agent_action = self.sess.run(self.network.sample_action, feed_dict={self.network.state: E.states})

        return (agent_action, None, None, None)

    def add_experiences(self, info, next_info, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        info_P = info[self.brain_to_imitate]
        next_info_P = next_info[self.brain_to_imitate]
        for agent_id in info_P.agents:
            if agent_id in next_info_P.agents:
                idx = info_P.agents.index(agent_id)
                next_idx = next_info_P.agents.index(agent_id)
                if not info_P.local_done[idx]:
                    self.training_buffer[agent_id]['states'].append(info_P.states[idx])
                    self.training_buffer[agent_id]['actions'].append(next_info_P.previous_actions[next_idx])
                    # self.training_buffer[agent_id]['rewards'].append(next_info.rewards[next_idx])

        info_E = next_info[self.brain_name]
        next_info_E = next_info[self.brain_name]
        for agent_id in info_E.agents:
            idx = info_E.agents.index(agent_id)
            next_idx = next_info_E.agents.index(agent_id)
            if not info_E.local_done[idx]:
                if agent_id not in self.cumulative_rewards:
                    self.cumulative_rewards[agent_id] = 0
                self.cumulative_rewards[agent_id] += next_info_E.rewards[next_idx]
                if agent_id not in self.episode_steps:
                    self.episode_steps[agent_id] = 0
                self.episode_steps[agent_id] += 1

    def process_experiences(self, info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        """

        info_P = info[self.brain_to_imitate]
        for l in range(len(info_P.agents)):
            if ((info_P.local_done[l] or 
                    len(self.training_buffer[info_P.agents[l]]['actions']) > self.trainer_parameters['time_horizon']) 
                    and len(self.training_buffer[info_P.agents[l]]['actions']) > 0):
                agent_id = info_P.agents[l]
                self.training_buffer.append_update_buffer(agent_id, 
                    batch_size = None, training_length=None)
                self.training_buffer[agent_id].reset_agent()

        info_E = info[self.brain_name]
        for l in range(len(info_E.agents)):
            if info_E.local_done[l]:
                agent_id = info_E.agents[l]
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
        Returns wether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        return len(self.training_buffer.update_buffer['actions']) > self.batch_size

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        batch_size = self.trainer_parameters['batch_size']


        self.training_buffer.update_buffer.shuffle()
        batch_losses = []
        for j in range(min(len(self.training_buffer.update_buffer['actions']) // self.batch_size, self.batches_per_epoch)):
            _buffer = self.training_buffer.update_buffer
            batch_states = np.array(_buffer['states'][j * batch_size:(j + 1) * batch_size])
            batch_actions = np.array(_buffer['actions'][j * batch_size:(j + 1) * batch_size])
            if not self.is_continuous:
                feed_dict = {
                    self.network.state: batch_states.reshape([-1, 1]), 
                    self.network.true_action: np.reshape(batch_actions, -1)
                }
            else:
                feed_dict = {
                    self.network.state: batch_states.reshape([self.batch_size, -1]), 
                    self.network.true_action: batch_actions.reshape([self.batch_size, -1])
                }
            loss, _ = self.sess.run([self.network.loss, self.network.update], feed_dict=feed_dict)
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





import numpy as np
import tensorflow as tf
import re
import os

from ppo.buffer import *
from ppo.models import *
import logging
logger = logging.getLogger("unityagents")

class Trainer(object):
    def __init__(self, sess, env, brain_name, trainer_parameters, training):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        #TODO: Might want to have multiple models here : Store them in a list ?
        with tf.variable_scope(re.sub('[^0-9a-zA-Z]+', '-', brain_name)):
            self.model = create_agent_model(env.brains[brain_name], 
                   lr=trainer_parameters['learning_rate'],
                   h_size=trainer_parameters['hidden_units'],
                   epsilon=trainer_parameters['epsilon'],
                   beta=trainer_parameters['beta'], 
                   max_step=trainer_parameters['max_steps'],
                   normalize=trainer_parameters['normalize'],
                   num_layers=trainer_parameters['num_layers'])
        self.sess = sess
        stats = {'cumulative_reward': [], 'episode_length': [], 'value_estimate': [],
                 'entropy': [], 'value_loss': [], 'policy_loss': [], 'learning_rate': []}
        self.stats = stats
        self.is_training = training
        self.use_normalize = trainer_parameters['normalize']
        # TODO: Figure out if using this buffer makes sense
        self.training_buffer = Buffer()
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.is_continuous = (env.brains[brain_name].action_space_type == "continuous")
        self.use_observations = (env.brains[brain_name].number_observations > 0)
        self.use_states = (env.brains[brain_name].state_space_size > 0)

        self.summary_path = './summaries/{}'.format(trainer_parameters['run_path']+'_'+brain_name)
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        #TODO: Some of these fields could be stored in a more efficient way
        self.summary_writer = tf.summary.FileWriter(self.summary_path)
        self.max_step = trainer_parameters['max_steps']
        self.buffer_size = trainer_parameters['buffer_size']
        self.batch_size = trainer_parameters['batch_size']
        self.num_epoch = trainer_parameters['num_epoch']
        self.summary_freq = trainer_parameters['summary_freq']
        self.brain_name = brain_name

    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.sess.run(self.model.global_step)

    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.sess.run(self.model.last_reward)

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        # TODO: Put this into the take_action method
        self.sess.run(self.model.increment_step)

    def update_last_reward(self):
        """
        Updates the last reward
        """
        # TODO: Put this into the take_action method
        if len(self.stats['cumulative_reward']) > 0:
            mean_reward = np.mean(self.stats['cumulative_reward'])
            self.sess.run(self.model.update_reward, feed_dict={self.model.new_reward: mean_reward})
            last_reward = self.sess.run(self.model.last_reward)

    def running_average(self, data, steps, running_mean, running_variance):
        """
        Computes new running mean and variances.
        :param data: New piece of data.
        :param steps: Total number of data so far.
        :param running_mean: TF op corresponding to stored running mean.
        :param running_variance: TF op corresponding to stored running variance.
        :return: New mean and variance values.
        """
        mean, var = self.sess.run([running_mean, running_variance])
        current_x = np.mean(data, axis=0)
        new_mean = mean + (current_x - mean) / (steps + 1)
        new_variance = var + (current_x - new_mean) * (current_x - mean)
        return new_mean, new_variance

    def take_action(self, info, env, brain_name):
        #This is not where these arguments should be
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :param env: Environment to take actions in.
        :param brain_name: Name of brain we are learning model for.
        :return: a tupple containing action, memories, values and an object
        to be passed to add experiences
        """
        steps = self.get_step()
        epsi = None
        feed_dict = {self.model.batch_size: len(info.states)}
        run_list = [self.model.output, self.model.probs, self.model.value, self.model.entropy,
                    self.model.learning_rate]
        if self.is_continuous:
            epsi = np.random.randn(len(info.states), env.brains[brain_name].action_space_size)
            feed_dict[self.model.epsilon] = epsi
        if self.use_observations:
            feed_dict[self.model.observation_in] = np.vstack(info.observations)
        if self.use_states:
            feed_dict[self.model.state_in] = info.states
        if self.is_training and env.brains[brain_name].state_space_type == "continuous" and self.use_states and self.use_normalize:
            new_mean, new_variance = self.running_average(info.states, steps, self.model.running_mean,
                                                          self.model.running_variance)
            feed_dict[self.model.new_mean] = new_mean
            feed_dict[self.model.new_variance] = new_variance
            run_list = run_list + [self.model.update_mean, self.model.update_variance]
            actions, a_dist, value, ent, learn_rate, _, _ = self.sess.run(run_list, feed_dict=feed_dict)
        else:
            actions, a_dist, value, ent, learn_rate = self.sess.run(run_list, feed_dict=feed_dict)
        self.stats['value_estimate'].append(value)
        self.stats['entropy'].append(ent)
        self.stats['learning_rate'].append(learn_rate)
        #This cannot be in trainer
        # new_info = env.step(actions, value={brain_name: value})[brain_name]
        # self.add_experiences(info, new_info, epsi, actions, a_dist, value)
        memories = None
        return (actions, memories, value, (actions, epsi, a_dist, value))

    def add_experiences(self, info, next_info, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        actions, epsi, a_dist, value = take_action_outputs
        for agent_id in info.agents:
            if agent_id in next_info.agents:
                idx = info.agents.index(agent_id)
                next_idx = idx = next_info.agents.index(agent_id)
                if not info.local_done[idx]:
                    if self.use_observations:
                        self.training_buffer[agent_id]['observations'].append_element([info.observations[0][idx]])
                    if self.use_states:
                        self.training_buffer[agent_id]['states'].append_element(info.states[idx])
                    if self.is_continuous:
                        self.training_buffer[agent_id]['epsilons'].append_element(epsi[idx])
                    self.training_buffer[agent_id]['actions'].append_element(actions[idx])
                    self.training_buffer[agent_id]['rewards'].append_element(next_info.rewards[next_idx])
                    self.training_buffer[agent_id]['action_probs'].append_element(a_dist[idx])
                    self.training_buffer[agent_id]['value_estimates'].append_element(value[idx][0])
                    if agent_id not in self.cumulative_rewards:
                        self.cumulative_rewards[agent_id] = 0
                    self.cumulative_rewards[agent_id] += next_info.rewards[next_idx]
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1

    def process_experiences(self, info, time_horizon, gamma, lambd):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        :param time_horizon: Max steps for individual agent history before processing.
        :param gamma: Discount factor.
        :param lambd: GAE factor.
        """
        for l in range(len(info.agents)):
            if (info.local_done[l] or len(self.training_buffer[info.agents[l]]['actions']) > time_horizon) and len(
                    self.training_buffer[info.agents[l]]['actions']) > 0:
                if info.local_done[l]:
                    value_next = 0.0
                else:
                    feed_dict = {self.model.batch_size: len(info.states)}
                    if self.use_observations:
                        feed_dict[self.model.observation_in] = np.vstack(info.observations)
                    if self.use_states:
                        feed_dict[self.model.state_in] = info.states
                    value_next = self.sess.run(self.model.value, feed_dict)[l]
                agent_id = info.agents[l]
                self.training_buffer[agent_id]['advantages'].set(
                    get_gae(
                        rewards=self.training_buffer[agent_id]['rewards'].get_batch(),
                        value_estimates=self.training_buffer[agent_id]['value_estimates'].get_batch(),
                        value_next=value_next, gamma=gamma, lambd=lambd)
                )
                self.training_buffer[agent_id]['discounted_returns'].set( \
                 self.training_buffer[agent_id]['advantages'].get_batch() \
                 + self.training_buffer[agent_id]['value_estimates'].get_batch())
                #TODO: Figure out if this way of using the buffer makes sense
                try:
                    self.training_buffer.append_global(agent_id)
                except:
                    print(self.training_buffer)
                    raise
                self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
                    self.stats['cumulative_reward'].append(self.cumulative_rewards[agent_id])
                    self.stats['episode_length'].append(self.episode_steps[agent_id])
                    self.cumulative_rewards[agent_id] = 0
                    self.episode_steps[agent_id] = 0


    def reset_buffers(self):
        """
        A signal that the buffer must be reset. Get typically called when the academy resets.
        """
        self.training_buffer.reset_all()
        for agent_id in self.cumulative_rewards:
            self.cumulative_rewards[agent_id] = 0
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0

    # IsReadyForUpdate(self):
    def is_ready_update(self):
        """
        Returns wether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        #TODO: Put this in update_model()
        return len(self.training_buffer.global_buffer['actions']) > self.buffer_size

    # def reset_buffers(self, brain_info=None, total=False):
    #     """
    #     Resets either all training buffers or local training buffers
    #     :param brain_info: The BrainInfo object containing agent ids.
    #     :param total: Whether to completely clear buffer.
    #     """
    #     if not total:
    #         for key in self.history_dict:
    #             self.history_dict[key] = empty_local_history(self.history_dict[key])
    #     else:
    #         self.history_dict = empty_all_history(agent_info=brain_info)

    def update_model(self):
        """
        Uses training_buffer to update model.
        :param batch_size: Size of each mini-batch update.
        :param num_epoch: How many passes through data to update model for.
        """
        num_epoch = self.num_epoch
        batch_size = self.batch_size
        total_v, total_p = 0, 0
        advantages = self.training_buffer.global_buffer['advantages'].get_batch()
        self.training_buffer.global_buffer['advantages'].set(
           (advantages - advantages.mean()) / advantages.std())
        for k in range(num_epoch):
            # training_buffer = shuffle_buffer(self.training_buffer)
            self.training_buffer.global_buffer.shuffle()
            for l in range(len(self.training_buffer.global_buffer['actions']) // batch_size):
                start = l * batch_size
                end = (l + 1) * batch_size
                feed_dict = {self.model.returns_holder: self.training_buffer.global_buffer['discounted_returns'][start:end],
                             self.model.advantage: np.vstack(self.training_buffer.global_buffer['advantages'][start:end]),
                             self.model.old_probs: np.vstack(self.training_buffer.global_buffer['action_probs'][start:end])}
                if self.is_continuous:
                    feed_dict[self.model.epsilon] = np.vstack(self.training_buffer.global_buffer['epsilons'][start:end])
                else:
                    feed_dict[self.model.action_holder] = np.hstack(self.training_buffer.global_buffer['actions'][start:end])
                if self.use_states:
                    feed_dict[self.model.state_in] = np.vstack(self.training_buffer.global_buffer['states'][start:end])
                if self.use_observations:
                    feed_dict[self.model.observation_in] = np.vstack(self.training_buffer.global_buffer['observations'][start:end])
                v_loss, p_loss, _ = self.sess.run([self.model.value_loss, self.model.policy_loss,
                                                   self.model.update_batch], feed_dict=feed_dict)
                total_v += v_loss
                total_p += p_loss
        self.stats['value_loss'].append(total_v)
        self.stats['policy_loss'].append(total_p)
        self.training_buffer.reset_global()

    def write_summary(self, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param lesson_number: The lesson the trainer is at.
        """
        #TODO: Is there a way to remove lesson_number of make it a flexible structure?
        if self.get_step() % self.summary_freq == 0 and self.get_step() != 0 and self.is_training:
            steps = self.get_step()
            if len(self.stats['cumulative_reward']) > 0:
                mean_reward = np.mean(self.stats['cumulative_reward'])
                print("{0} : Step: {1}. Mean Reward: {2}. Std of Reward: {3}."
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

    def write_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            s_op = tf.summary.text(key,
                    tf.convert_to_tensor(([[str(x), str(input_dict[x])] for x in input_dict]))
                    )
            s = self.sess.run(s_op)
            self.summary_writer.add_summary(s, self.get_step())
        except:
            logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
            pass



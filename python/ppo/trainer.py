import numpy as np
import tensorflow as tf

from ppo.history import *


class Trainer(object):
    def __init__(self, ppo_model, sess, info, is_continuous, use_observations, use_states, training):
        """
        Responsible for collecting experiences and training PPO model.
        :param ppo_model: Tensorflow graph defining model.
        :param sess: Tensorflow session.
        :param info: Environment BrainInfo object.
        :param is_continuous: Whether action-space is continuous.
        :param use_observations: Whether agent takes image observations.
        """
        self.model = ppo_model
        self.sess = sess
        stats = {'cumulative_reward': [], 'episode_length': [], 'value_estimate': [],
                 'entropy': [], 'value_loss': [], 'policy_loss': [], 'learning_rate': []}
        self.stats = stats
        self.is_training = training
        self.reset_buffers(info, total=True)
        self.training_buffer = vectorize_history(empty_local_history({}))
        self.is_continuous = is_continuous
        self.use_observations = use_observations
        self.use_states = use_states

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

    def take_action(self, info, env, brain_name, steps, normalize):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :param env: Environment to take actions in.
        :param brain_name: Name of brain we are learning model for.
        :return: BrainInfo corresponding to new environment state.
        """
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
        if self.is_training and env.brains[brain_name].state_space_type == "continuous" and self.use_states and normalize:
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
        new_info = env.step(actions, value={brain_name: value})[brain_name]
        self.add_experiences(info, new_info, epsi, actions, a_dist, value)
        return new_info

    def add_experiences(self, info, next_info, epsi, actions, a_dist, value):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param epsi: Epsilon value (for continuous control)
        :param actions: Chosen actions.
        :param a_dist: Action probabilities.
        :param value: Value estimates.
        """
        for (agent, history) in self.history_dict.items():
            if agent in info.agents:
                idx = info.agents.index(agent)
                if not info.local_done[idx]:
                    if self.use_observations:
                        history['observations'].append([info.observations[0][idx]])
                    if self.use_states:
                        history['states'].append(info.states[idx])
                    if self.is_continuous:
                        history['epsilons'].append(epsi[idx])
                    history['actions'].append(actions[idx])
                    history['rewards'].append(next_info.rewards[idx])
                    history['action_probs'].append(a_dist[idx])
                    history['value_estimates'].append(value[idx][0])
                    history['cumulative_reward'] += next_info.rewards[idx]
                    history['episode_steps'] += 1

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
            if (info.local_done[l] or len(self.history_dict[info.agents[l]]['actions']) > time_horizon) and len(
                    self.history_dict[info.agents[l]]['actions']) > 0:
                if info.local_done[l]:
                    value_next = 0.0
                else:
                    feed_dict = {self.model.batch_size: len(info.states)}
                    if self.use_observations:
                        feed_dict[self.model.observation_in] = np.vstack(info.observations)
                    if self.use_states:
                        feed_dict[self.model.state_in] = info.states
                    value_next = self.sess.run(self.model.value, feed_dict)[l]
                history = vectorize_history(self.history_dict[info.agents[l]])
                history['advantages'] = get_gae(rewards=history['rewards'],
                                                value_estimates=history['value_estimates'],
                                                value_next=value_next, gamma=gamma, lambd=lambd)
                history['discounted_returns'] = history['advantages'] + history['value_estimates']
                if len(self.training_buffer['actions']) > 0:
                    append_history(global_buffer=self.training_buffer, local_buffer=history)
                else:
                    set_history(global_buffer=self.training_buffer, local_buffer=history)
                self.history_dict[info.agents[l]] = empty_local_history(self.history_dict[info.agents[l]])
                if info.local_done[l]:
                    self.stats['cumulative_reward'].append(history['cumulative_reward'])
                    self.stats['episode_length'].append(history['episode_steps'])
                    history['cumulative_reward'] = 0
                    history['episode_steps'] = 0

    def reset_buffers(self, brain_info=None, total=False):
        """
        Resets either all training buffers or local training buffers
        :param brain_info: The BrainInfo object containing agent ids.
        :param total: Whether to completely clear buffer.
        """
        if not total:
            for key in self.history_dict:
                self.history_dict[key] = empty_local_history(self.history_dict[key])
        else:
            self.history_dict = empty_all_history(agent_info=brain_info)

    def update_model(self, batch_size, num_epoch):
        """
        Uses training_buffer to update model.
        :param batch_size: Size of each mini-batch update.
        :param num_epoch: How many passes through data to update model for.
        """
        total_v, total_p = 0, 0
        advantages = self.training_buffer['advantages']
        self.training_buffer['advantages'] = (advantages - advantages.mean()) / advantages.std()
        for k in range(num_epoch):
            training_buffer = shuffle_buffer(self.training_buffer)
            for l in range(len(training_buffer['actions']) // batch_size):
                start = l * batch_size
                end = (l + 1) * batch_size
                feed_dict = {self.model.returns_holder: training_buffer['discounted_returns'][start:end],
                             self.model.advantage: np.vstack(training_buffer['advantages'][start:end]),
                             self.model.old_probs: np.vstack(training_buffer['action_probs'][start:end])}
                if self.is_continuous:
                    feed_dict[self.model.epsilon] = np.vstack(training_buffer['epsilons'][start:end])
                else:
                    feed_dict[self.model.action_holder] = np.hstack(training_buffer['actions'][start:end])
                if self.use_states:
                    feed_dict[self.model.state_in] = np.vstack(training_buffer['states'][start:end])
                if self.use_observations:
                    feed_dict[self.model.observation_in] = np.vstack(training_buffer['observations'][start:end])
                v_loss, p_loss, _ = self.sess.run([self.model.value_loss, self.model.policy_loss,
                                                   self.model.update_batch], feed_dict=feed_dict)
                total_v += v_loss
                total_p += p_loss
        self.stats['value_loss'].append(total_v)
        self.stats['policy_loss'].append(total_p)
        self.training_buffer = vectorize_history(empty_local_history({}))

    def write_summary(self, summary_writer, steps, lesson_number):
        """
        Saves training statistics to Tensorboard.
        :param summary_writer: writer associated with Tensorflow session.
        :param steps: Number of environment steps in training process.
        """
        if len(self.stats['cumulative_reward']) > 0:
            mean_reward = np.mean(self.stats['cumulative_reward'])
            print("Step: {0}. Mean Reward: {1}. Std of Reward: {2}."
                  .format(steps, mean_reward, np.std(self.stats['cumulative_reward'])))
        summary = tf.Summary()
        for key in self.stats:
            if len(self.stats[key]) > 0:
                stat_mean = float(np.mean(self.stats[key]))
                summary.value.add(tag='Info/{}'.format(key), simple_value=stat_mean)
                self.stats[key] = []
        summary.value.add(tag='Info/Lesson', simple_value=lesson_number)
        summary_writer.add_summary(summary, steps)
        summary_writer.flush()

    def write_text(self, summary_writer, key, input_dict, steps):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param summary_writer: writer associated with Tensorflow session.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        :param steps: Number of environment steps in training process.
        """
        try:
            s_op = tf.summary.text(key,
                    tf.convert_to_tensor(([[str(x), str(input_dict[x])] for x in input_dict]))
                    )
            s = self.sess.run(s_op)
            summary_writer.add_summary(s, steps)
        except:
            print("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")
            pass



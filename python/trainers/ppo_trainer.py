# # Unity ML Agents
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described [here](https://arxiv.org/abs/1707.06347).

import logging
import os

import numpy as np
import tensorflow as tf

from trainers.buffer import Buffer
from trainers.ppo_models import *
from trainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("unityagents")


class PPOTrainer(Trainer):
    """The PPOTrainer is an implementation of the PPO algorythm."""

    def __init__(self, sess, env, brain_name, trainer_parameters, training):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.param_keys = ['batch_size', 'beta', 'buffer_size', 'epsilon', 'gamma', 'hidden_units', 'lambd',
                           'learning_rate',
                           'max_steps', 'normalize', 'num_epoch', 'num_layers', 'time_horizon', 'sequence_length',
                           'summary_freq',
                           'use_recurrent', 'graph_scope', 'summary_path']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the PPO trainer of "
                                            "brain {1}.".format(k, brain_name))

        super(PPOTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.sequence_length = 1
        self.m_size = None
        if self.use_recurrent:
            self.m_size = env.brains[brain_name].memory_space_size
            self.sequence_length = trainer_parameters["sequence_length"]
        if self.use_recurrent:
            if (self.m_size == 0):
                raise UnityTrainerException("The memory size for brain {0} is 0 even though the trainer uses recurrent."
                                            .format(brain_name))
            elif (self.m_size % 4 != 0):
                raise UnityTrainerException("The memory size for brain {0} is {1} but it must be divisible by 4."
                                            .format(brain_name, self.m_size))

        self.variable_scope = trainer_parameters['graph_scope']
        with tf.variable_scope(self.variable_scope):
            self.model = create_agent_model(env.brains[brain_name],
                                            lr=float(trainer_parameters['learning_rate']),
                                            h_size=int(trainer_parameters['hidden_units']),
                                            epsilon=float(trainer_parameters['epsilon']),
                                            beta=float(trainer_parameters['beta']),
                                            max_step=float(trainer_parameters['max_steps']),
                                            normalize=trainer_parameters['normalize'],
                                            use_recurrent=trainer_parameters['use_recurrent'],
                                            num_layers=int(trainer_parameters['num_layers']),
                                            m_size=self.m_size)

        stats = {'cumulative_reward': [], 'episode_length': [], 'value_estimate': [],
                 'entropy': [], 'value_loss': [], 'policy_loss': [], 'learning_rate': []}
        self.stats = stats

        self.training_buffer = Buffer()
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.is_continuous = (env.brains[brain_name].action_space_type == "continuous")
        self.use_observations = (env.brains[brain_name].number_observations > 0)
        self.use_states = (env.brains[brain_name].state_space_size > 0)
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)

    def __str__(self):
        return '''Hypermarameters for the PPO Trainer of brain {0}: \n{1}'''.format(
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
        return self.sess.run(self.model.last_reward)

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.sess.run(self.model.increment_step)

    def update_last_reward(self):
        """
        Updates the last reward
        """
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

    def take_action(self, info):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param info: Current BrainInfo from environment.
        :return: a tupple containing action, memories, values and an object
        to be passed to add experiences
        """
        steps = self.get_step
        info = info[self.brain_name]
        epsi = None
        feed_dict = {self.model.batch_size: len(info.states), self.model.sequence_length: 1}
        run_list = [self.model.output, self.model.probs, self.model.value, self.model.entropy,
                    self.model.learning_rate]
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
        if (self.is_training and self.brain.state_space_type == "continuous" and
                self.use_states and self.trainer_parameters['normalize']):
            new_mean, new_variance = self.running_average(info.states, steps, self.model.running_mean,
                                                          self.model.running_variance)
            feed_dict[self.model.new_mean] = new_mean
            feed_dict[self.model.new_variance] = new_variance
            run_list = run_list + [self.model.update_mean, self.model.update_variance]
            # only ask for memories if use_recurrent
            if self.use_recurrent:
                actions, a_dist, value, ent, learn_rate, memories, _, _ = self.sess.run(run_list, feed_dict=feed_dict)
            else:
                actions, a_dist, value, ent, learn_rate, _, _ = self.sess.run(run_list, feed_dict=feed_dict)
                memories = None
        else:
            if self.use_recurrent:
                actions, a_dist, value, ent, learn_rate, memories = self.sess.run(run_list, feed_dict=feed_dict)
            else:
                actions, a_dist, value, ent, learn_rate = self.sess.run(run_list, feed_dict=feed_dict)
                memories = None
        self.stats['value_estimate'].append(value)
        self.stats['entropy'].append(ent)
        self.stats['learning_rate'].append(learn_rate)
        return actions, memories, value, (actions, epsi, a_dist, value)

    def add_experiences(self, info, next_info, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param info: Current BrainInfo.
        :param next_info: Next BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        info = info[self.brain_name]
        next_info = next_info[self.brain_name]
        actions, epsi, a_dist, value = take_action_outputs
        for agent_id in info.agents:
            if agent_id in next_info.agents:
                idx = info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not info.local_done[idx]:
                    if self.use_observations:
                        for i, _ in enumerate(info.observations):
                            self.training_buffer[agent_id]['observations%d' % i].append(info.observations[i][idx])
                    if self.use_states:
                        self.training_buffer[agent_id]['states'].append(info.states[idx])
                    if self.use_recurrent:
                        self.training_buffer[agent_id]['memory'].append(info.memories[idx])
                    if self.is_continuous:
                        self.training_buffer[agent_id]['epsilons'].append(epsi[idx])
                    self.training_buffer[agent_id]['actions'].append(actions[idx])
                    self.training_buffer[agent_id]['rewards'].append(next_info.rewards[next_idx])
                    self.training_buffer[agent_id]['action_probs'].append(a_dist[idx])
                    self.training_buffer[agent_id]['value_estimates'].append(value[idx][0])
                    if agent_id not in self.cumulative_rewards:
                        self.cumulative_rewards[agent_id] = 0
                    self.cumulative_rewards[agent_id] += next_info.rewards[next_idx]
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1

    def process_experiences(self, info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param info: Current BrainInfo
        """

        info = info[self.brain_name]
        for l in range(len(info.agents)):
            agent_actions = self.training_buffer[info.agents[l]]['actions']
            if ((info.local_done[l] or len(agent_actions) > self.trainer_parameters['time_horizon'])
                    and len(agent_actions) > 0):

                if info.local_done[l]:
                    value_next = 0.0
                else:
                    feed_dict = {self.model.batch_size: len(info.states), self.model.sequence_length: 1}
                    if self.use_observations:
                        for i in range(len(info.observations)):
                            feed_dict[self.model.observation_in[i]] = info.observations[i]
                    if self.use_states:
                        feed_dict[self.model.state_in] = info.states
                    if self.use_recurrent:
                        feed_dict[self.model.memory_in] = info.memories
                    value_next = self.sess.run(self.model.value, feed_dict)[l]
                agent_id = info.agents[l]
                self.training_buffer[agent_id]['advantages'].set(
                    get_gae(
                        rewards=self.training_buffer[agent_id]['rewards'].get_batch(),
                        value_estimates=self.training_buffer[agent_id]['value_estimates'].get_batch(),
                        value_next=value_next,
                        gamma=self.trainer_parameters['gamma'],
                        lambd=self.trainer_parameters['lambd'])
                )
                self.training_buffer[agent_id]['discounted_returns'].set( \
                    self.training_buffer[agent_id]['advantages'].get_batch() \
                    + self.training_buffer[agent_id]['value_estimates'].get_batch())

                self.training_buffer.append_update_buffer(agent_id,
                                                          batch_size=None, training_length=self.sequence_length)

                self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
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
        return len(self.training_buffer.update_buffer['actions']) > self.trainer_parameters['buffer_size']

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        num_epoch = self.trainer_parameters['num_epoch']
        batch_size = self.trainer_parameters['batch_size']
        total_v, total_p = 0, 0
        advantages = self.training_buffer.update_buffer['advantages'].get_batch()
        self.training_buffer.update_buffer['advantages'].set(
            (advantages - advantages.mean()) / advantages.std())
        for k in range(num_epoch):
            self.training_buffer.update_buffer.shuffle()
            for l in range(len(self.training_buffer.update_buffer['actions']) // batch_size):
                start = l * batch_size
                end = (l + 1) * batch_size
                _buffer = self.training_buffer.update_buffer
                feed_dict = {self.model.batch_size: batch_size,
                             self.model.sequence_length: self.sequence_length,
                             self.model.returns_holder: np.array(_buffer['discounted_returns'][start:end]).reshape(
                                 [-1]),
                             self.model.advantage: np.array(_buffer['advantages'][start:end]).reshape([-1, 1]),
                             self.model.old_probs: np.array(
                                 _buffer['action_probs'][start:end]).reshape([-1, self.brain.action_space_size])}
                if self.is_continuous:
                    feed_dict[self.model.epsilon] = np.array(
                        _buffer['epsilons'][start:end]).reshape([-1, self.brain.action_space_size])
                else:
                    feed_dict[self.model.action_holder] = np.array(
                        _buffer['actions'][start:end]).reshape([-1])
                if self.use_states:
                    if self.brain.state_space_type == "continuous":
                        feed_dict[self.model.state_in] = np.array(
                            _buffer['states'][start:end]).reshape(
                            [-1, self.brain.state_space_size * self.brain.stacked_states])
                    else:
                        feed_dict[self.model.state_in] = np.array(
                            _buffer['states'][start:end]).reshape([-1, 1])
                if self.use_observations:
                    for i, _ in enumerate(self.model.observation_in):
                        _obs = np.array(_buffer['observations%d' % i][start:end])
                        (_batch, _seq, _w, _h, _c) = _obs.shape
                        feed_dict[self.model.observation_in[i]] = _obs.reshape([-1, _w, _h, _c])
                # Memories are zeros
                if self.use_recurrent:
                    feed_dict[self.model.memory_in] = np.zeros([batch_size, self.m_size])
                v_loss, p_loss, _ = self.sess.run([self.model.value_loss, self.model.policy_loss,
                                                   self.model.update_batch], feed_dict=feed_dict)
                total_v += v_loss
                total_p += p_loss
        self.stats['value_loss'].append(total_v)
        self.stats['policy_loss'].append(total_p)
        self.training_buffer.reset_update_buffer()

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
                logger.info(" {}: Step: {}. Mean Reward: {:0.3f}. Std of Reward: {:0.3f}."
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


def discount_rewards(r, gamma=0.99, value_next=0.0):
    """
    Computes discounted sum of future rewards for use in updating value estimate.
    :param r: List of rewards.
    :param gamma: Discount factor.
    :param value_next: T+1 value estimate for returns calculation.
    :return: discounted sum of future rewards as list.
    """
    discounted_r = np.zeros_like(r)
    running_add = value_next
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def get_gae(rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
    """
    Computes generalized advantage estimate for use in updating policy.
    :param rewards: list of rewards for time-steps t to T.
    :param value_next: Value estimate for time-step T+1.
    :param value_estimates: list of value estimates for time-steps t to T.
    :param gamma: Discount factor.
    :param lambd: GAE weighing factor.
    :return: list of advantage estimates for time-steps t to T.
    """
    value_estimates = np.asarray(value_estimates.tolist() + [value_next])
    delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
    advantage = discount_rewards(r=delta_t, gamma=gamma * lambd)
    return advantage

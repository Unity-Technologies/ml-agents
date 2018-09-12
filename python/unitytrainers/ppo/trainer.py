# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described (https://arxiv.org/abs/1707.06347).

import logging
import os

import numpy as np
import tensorflow as tf

from unityagents import AllBrainInfo, BrainInfo
from unitytrainers.buffer import Buffer
from unitytrainers.ppo.models import PPOModel
from unitytrainers.trainer import UnityTrainerException, Trainer

logger = logging.getLogger("unityagents")


class PPOTrainer(Trainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(self, sess, env, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training PPO model.
        :param sess: Tensorflow session.
        :param env: The UnityEnvironment.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        """
        self.param_keys = ['batch_size', 'beta', 'buffer_size', 'epsilon', 'gamma', 'hidden_units', 'lambd',
                           'learning_rate', 'max_steps', 'normalize', 'num_epoch', 'num_layers',
                           'time_horizon', 'sequence_length', 'summary_freq', 'use_recurrent',
                           'graph_scope', 'summary_path', 'memory_size', 'use_curiosity', 'curiosity_strength',
                           'curiosity_enc_size']

        for k in self.param_keys:
            if k not in trainer_parameters:
                raise UnityTrainerException("The hyperparameter {0} could not be found for the PPO trainer of "
                                            "brain {1}.".format(k, brain_name))

        super(PPOTrainer, self).__init__(sess, env, brain_name, trainer_parameters, training)

        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.use_curiosity = bool(trainer_parameters['use_curiosity'])
        self.sequence_length = 1
        self.step = 0
        self.has_updated = False
        self.m_size = None
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityTrainerException("The memory size for brain {0} is 0 even though the trainer uses recurrent."
                                            .format(brain_name))
            elif self.m_size % 4 != 0:
                raise UnityTrainerException("The memory size for brain {0} is {1} but it must be divisible by 4."
                                            .format(brain_name, self.m_size))

        self.variable_scope = trainer_parameters['graph_scope']
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

        stats = {'cumulative_reward': [], 'episode_length': [], 'value_estimate': [],
                 'entropy': [], 'value_loss': [], 'policy_loss': [], 'learning_rate': []}
        if self.use_curiosity:
            stats['forward_loss'] = []
            stats['inverse_loss'] = []
            stats['intrinsic_reward'] = []
            self.intrinsic_rewards = {}
        self.stats = stats

        self.training_buffer = Buffer()
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.is_continuous_action = (env.brains[brain_name].vector_action_space_type == "continuous")
        self.is_continuous_observation = (env.brains[brain_name].vector_observation_space_type == "continuous")
        self.use_visual_obs = (env.brains[brain_name].number_visual_observations > 0)
        self.use_vector_obs = (env.brains[brain_name].vector_observation_space_size > 0)
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)

        self.inference_run_list = [self.model.output, self.model.all_probs, self.model.value,
                                   self.model.entropy, self.model.learning_rate]
        if self.is_continuous_action:
            self.inference_run_list.append(self.model.output_pre)
        if self.use_recurrent:
            self.inference_run_list.extend([self.model.memory_out])
        if (self.is_training and self.is_continuous_observation and
                self.use_vector_obs and self.trainer_parameters['normalize']):
            self.inference_run_list.extend([self.model.update_mean, self.model.update_variance])

    def __str__(self):
        return '''Hyperparameters for the PPO Trainer of brain {0}: \n{1}'''.format(
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
        return self.sess.run(self.model.last_reward)

    def increment_step_and_update_last_reward(self):
        """
        Increment the step count of the trainer and Updates the last reward
        """
        if len(self.stats['cumulative_reward']) > 0:
            mean_reward = np.mean(self.stats['cumulative_reward'])
            self.sess.run([self.model.update_reward,
                           self.model.increment_step],
                          feed_dict={self.model.new_reward: mean_reward})
        else:
            self.sess.run(self.model.increment_step)
        self.step = self.sess.run(self.model.global_step)

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions given observations information, and takes them in environment.
        :param all_brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        curr_brain_info = all_brain_info[self.brain_name]
        if len(curr_brain_info.agents) == 0:
            return [], [], [], None

        feed_dict = {self.model.batch_size: len(curr_brain_info.vector_observations),
                     self.model.sequence_length: 1}
        if self.use_recurrent:
            if not self.is_continuous_action:
                feed_dict[self.model.prev_action] = curr_brain_info.previous_vector_actions.flatten()
            if curr_brain_info.memories.shape[1] == 0:
                curr_brain_info.memories = np.zeros((len(curr_brain_info.agents), self.m_size))
            feed_dict[self.model.memory_in] = curr_brain_info.memories
        if self.use_visual_obs:
            for i, _ in enumerate(curr_brain_info.visual_observations):
                feed_dict[self.model.visual_in[i]] = curr_brain_info.visual_observations[i]
        if self.use_vector_obs:
            feed_dict[self.model.vector_in] = curr_brain_info.vector_observations

        values = self.sess.run(self.inference_run_list, feed_dict=feed_dict)
        run_out = dict(zip(self.inference_run_list, values))

        self.stats['value_estimate'].append(run_out[self.model.value].mean())
        self.stats['entropy'].append(run_out[self.model.entropy].mean())
        self.stats['learning_rate'].append(run_out[self.model.learning_rate])
        if self.use_recurrent:
            return run_out[self.model.output], run_out[self.model.memory_out], None, run_out
        else:
            return run_out[self.model.output], None, None, run_out

    def construct_curr_info(self, next_info: BrainInfo) -> BrainInfo:
        """
        Constructs a BrainInfo which contains the most recent previous experiences for all agents info
        which correspond to the agents in a provided next_info.
        :BrainInfo next_info: A t+1 BrainInfo.
        :return: curr_info: Reconstructed BrainInfo to match agents of next_info.
        """
        visual_observations = [[]]
        vector_observations = []
        text_observations = []
        memories = []
        rewards = []
        local_dones = []
        max_reacheds = []
        agents = []
        prev_vector_actions = []
        prev_text_actions = []
        for agent_id in next_info.agents:
            agent_brain_info = self.training_buffer[agent_id].last_brain_info
            agent_index = agent_brain_info.agents.index(agent_id)
            if agent_brain_info is None:
                agent_brain_info = next_info
            for i in range(len(next_info.visual_observations)):
                visual_observations[i].append(agent_brain_info.visual_observations[i][agent_index])
            vector_observations.append(agent_brain_info.vector_observations[agent_index])
            text_observations.append(agent_brain_info.text_observations[agent_index])
            if self.use_recurrent:
                memories.append(agent_brain_info.memories[agent_index])
            rewards.append(agent_brain_info.rewards[agent_index])
            local_dones.append(agent_brain_info.local_done[agent_index])
            max_reacheds.append(agent_brain_info.max_reached[agent_index])
            agents.append(agent_brain_info.agents[agent_index])
            prev_vector_actions.append(agent_brain_info.previous_vector_actions[agent_index])
            prev_text_actions.append(agent_brain_info.previous_text_actions[agent_index])
        curr_info = BrainInfo(visual_observations, vector_observations, text_observations, memories, rewards,
                              agents, local_dones, prev_vector_actions, prev_text_actions, max_reacheds)
        return curr_info

    def generate_intrinsic_rewards(self, curr_info, next_info):
        """
        Generates intrinsic reward used for Curiosity-based training.
        :BrainInfo curr_info: Current BrainInfo.
        :BrainInfo next_info: Next BrainInfo.
        :return: Intrinsic rewards for all agents.
        """
        if self.use_curiosity:
            feed_dict = {self.model.batch_size: len(next_info.vector_observations), self.model.sequence_length: 1}
            if self.is_continuous_action:
                feed_dict[self.model.output] = next_info.previous_vector_actions
            else:
                feed_dict[self.model.action_holder] = next_info.previous_vector_actions.flatten()

            if curr_info.agents != next_info.agents:
                curr_info = self.construct_curr_info(next_info)

            if self.use_visual_obs:
                for i in range(len(curr_info.visual_observations)):
                    feed_dict[self.model.visual_in[i]] = curr_info.visual_observations[i]
                    feed_dict[self.model.next_visual_in[i]] = next_info.visual_observations[i]
            if self.use_vector_obs:
                feed_dict[self.model.vector_in] = curr_info.vector_observations
                feed_dict[self.model.next_vector_in] = next_info.vector_observations
            if self.use_recurrent:
                if curr_info.memories.shape[1] == 0:
                    curr_info.memories = np.zeros((len(curr_info.agents), self.m_size))
                feed_dict[self.model.memory_in] = curr_info.memories
            intrinsic_rewards = self.sess.run(self.model.intrinsic_reward,
                                              feed_dict=feed_dict) * float(self.has_updated)
            return intrinsic_rewards
        else:
            return None

    def generate_value_estimate(self, brain_info, idx):
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :return: Value estimate.
        """
        feed_dict = {self.model.batch_size: 1, self.model.sequence_length: 1}
        if self.use_visual_obs:
            for i in range(len(brain_info.visual_observations)):
                feed_dict[self.model.visual_in[i]] = [brain_info.visual_observations[i][idx]]
        if self.use_vector_obs:
            feed_dict[self.model.vector_in] = [brain_info.vector_observations[idx]]
        if self.use_recurrent:
            if brain_info.memories.shape[1] == 0:
                brain_info.memories = np.zeros(
                    (len(brain_info.vector_observations), self.m_size))
            feed_dict[self.model.memory_in] = [brain_info.memories[idx]]
        if not self.is_continuous_action and self.use_recurrent:
            feed_dict[self.model.prev_action] = brain_info.previous_vector_actions[idx].flatten()
        value_estimate = self.sess.run(self.model.value, feed_dict)
        return value_estimate

    def add_experiences(self, curr_all_info: AllBrainInfo, next_all_info: AllBrainInfo, take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the take action method.
        """
        curr_info = curr_all_info[self.brain_name]
        next_info = next_all_info[self.brain_name]

        for agent_id in curr_info.agents:
            self.training_buffer[agent_id].last_brain_info = curr_info
            self.training_buffer[agent_id].last_take_action_outputs = take_action_outputs

        intrinsic_rewards = self.generate_intrinsic_rewards(curr_info, next_info)

        for agent_id in next_info.agents:
            stored_info = self.training_buffer[agent_id].last_brain_info
            stored_take_action_outputs = self.training_buffer[agent_id].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    if self.use_visual_obs:
                        for i, _ in enumerate(stored_info.visual_observations):
                            self.training_buffer[agent_id]['visual_obs%d' % i].append(
                                stored_info.visual_observations[i][idx])
                            self.training_buffer[agent_id]['next_visual_obs%d' % i].append(
                                next_info.visual_observations[i][idx])
                    if self.use_vector_obs:
                        self.training_buffer[agent_id]['vector_obs'].append(stored_info.vector_observations[idx])
                        self.training_buffer[agent_id]['next_vector_in'].append(
                            next_info.vector_observations[next_idx])
                    if self.use_recurrent:
                        if stored_info.memories.shape[1] == 0:
                            stored_info.memories = np.zeros((len(stored_info.agents), self.m_size))
                        self.training_buffer[agent_id]['memory'].append(stored_info.memories[idx])
                    actions = stored_take_action_outputs[self.model.output]
                    if self.is_continuous_action:
                        actions_pre = stored_take_action_outputs[self.model.output_pre]
                        self.training_buffer[agent_id]['actions_pre'].append(actions_pre[idx])
                    a_dist = stored_take_action_outputs[self.model.all_probs]
                    value = stored_take_action_outputs[self.model.value]
                    self.training_buffer[agent_id]['actions'].append(actions[idx])
                    self.training_buffer[agent_id]['prev_action'].append(stored_info.previous_vector_actions[idx])
                    self.training_buffer[agent_id]['masks'].append(1.0)
                    if self.use_curiosity:
                        self.training_buffer[agent_id]['rewards'].append(next_info.rewards[next_idx] +
                                                                         intrinsic_rewards[next_idx])
                    else:
                        self.training_buffer[agent_id]['rewards'].append(next_info.rewards[next_idx])
                    self.training_buffer[agent_id]['action_probs'].append(a_dist[idx])
                    self.training_buffer[agent_id]['value_estimates'].append(value[idx][0])

                    if agent_id not in self.cumulative_rewards:
                        self.cumulative_rewards[agent_id] = 0
                    self.cumulative_rewards[agent_id] += next_info.rewards[next_idx]
                    if self.use_curiosity:
                        if agent_id not in self.intrinsic_rewards:
                            self.intrinsic_rewards[agent_id] = 0
                        self.intrinsic_rewards[agent_id] += intrinsic_rewards[next_idx]
                if not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1

    def process_experiences(self, current_info: AllBrainInfo, new_info: AllBrainInfo):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Dictionary of all current brains and corresponding BrainInfo.
        :param new_info: Dictionary of all next brains and corresponding BrainInfo.
        """

        info = new_info[self.brain_name]
        for l in range(len(info.agents)):
            agent_actions = self.training_buffer[info.agents[l]]['actions']
            if ((info.local_done[l] or len(agent_actions) > self.trainer_parameters['time_horizon'])
                    and len(agent_actions) > 0):
                agent_id = info.agents[l]
                if info.local_done[l] and not info.max_reached[l]:
                    value_next = 0.0
                else:
                    if info.max_reached[l]:
                        bootstrapping_info = self.training_buffer[agent_id].last_brain_info
                        idx = bootstrapping_info.agents.index(agent_id)
                    else:
                        bootstrapping_info = info
                        idx = l
                    value_next = self.generate_value_estimate(bootstrapping_info, idx)

                self.training_buffer[agent_id]['advantages'].set(
                    get_gae(
                        rewards=self.training_buffer[agent_id]['rewards'].get_batch(),
                        value_estimates=self.training_buffer[agent_id]['value_estimates'].get_batch(),
                        value_next=value_next,
                        gamma=self.trainer_parameters['gamma'],
                        lambd=self.trainer_parameters['lambd']))
                self.training_buffer[agent_id]['discounted_returns'].set(
                    self.training_buffer[agent_id]['advantages'].get_batch()
                    + self.training_buffer[agent_id]['value_estimates'].get_batch())

                self.training_buffer.append_update_buffer(agent_id, batch_size=None,
                                                          training_length=self.sequence_length)

                self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
                    self.stats['cumulative_reward'].append(
                        self.cumulative_rewards.get(agent_id, 0))
                    self.stats['episode_length'].append(
                        self.episode_steps.get(agent_id, 0))
                    self.cumulative_rewards[agent_id] = 0
                    self.episode_steps[agent_id] = 0
                    if self.use_curiosity:
                        self.stats['intrinsic_reward'].append(
                            self.intrinsic_rewards.get(agent_id, 0))
                        self.intrinsic_rewards[agent_id] = 0

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
        if self.use_curiosity:
            for agent_id in self.intrinsic_rewards:
                self.intrinsic_rewards[agent_id] = 0

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        size_of_buffer = len(self.training_buffer.update_buffer['actions'])
        return size_of_buffer > max(int(self.trainer_parameters['buffer_size'] / self.sequence_length), 1)

    def update_model(self):
        """
        Uses training_buffer to update model.
        """
        n_sequences = max(int(self.trainer_parameters['batch_size'] / self.sequence_length), 1)
        value_total, policy_total, forward_total, inverse_total = [], [], [], []
        advantages = self.training_buffer.update_buffer['advantages'].get_batch()
        self.training_buffer.update_buffer['advantages'].set(
            (advantages - advantages.mean()) / (advantages.std() + 1e-10))
        num_epoch = self.trainer_parameters['num_epoch']
        for k in range(num_epoch):
            self.training_buffer.update_buffer.shuffle()
            buffer = self.training_buffer.update_buffer
            for l in range(len(self.training_buffer.update_buffer['actions']) // n_sequences):
                start = l * n_sequences
                end = (l + 1) * n_sequences
                feed_dict = {self.model.batch_size: n_sequences,
                             self.model.sequence_length: self.sequence_length,
                             self.model.mask_input: np.array(buffer['masks'][start:end]).flatten(),
                             self.model.returns_holder: np.array(buffer['discounted_returns'][start:end]).flatten(),
                             self.model.old_value: np.array(buffer['value_estimates'][start:end]).flatten(),
                             self.model.advantage: np.array(buffer['advantages'][start:end]).reshape([-1, 1]),
                             self.model.all_old_probs: np.array(buffer['action_probs'][start:end]).reshape(
                                 [-1, self.brain.vector_action_space_size])}
                if self.is_continuous_action:
                    feed_dict[self.model.output_pre] = np.array(buffer['actions_pre'][start:end]).reshape(
                        [-1, self.brain.vector_action_space_size])
                else:
                    feed_dict[self.model.action_holder] = np.array(buffer['actions'][start:end]).flatten()
                    if self.use_recurrent:
                        feed_dict[self.model.prev_action] = np.array(buffer['prev_action'][start:end]).flatten()
                if self.use_vector_obs:
                    if self.is_continuous_observation:
                        total_observation_length = self.brain.vector_observation_space_size * \
                                                   self.brain.num_stacked_vector_observations
                        feed_dict[self.model.vector_in] = np.array(buffer['vector_obs'][start:end]).reshape(
                            [-1, total_observation_length])
                        if self.use_curiosity:
                            feed_dict[self.model.next_vector_in] = np.array(buffer['next_vector_in'][start:end]) \
                                .reshape([-1, total_observation_length])
                    else:
                        feed_dict[self.model.vector_in] = np.array(buffer['vector_obs'][start:end]).reshape(
                            [-1, self.brain.num_stacked_vector_observations])
                        if self.use_curiosity:
                            feed_dict[self.model.next_vector_in] = np.array(buffer['next_vector_in'][start:end]) \
                                .reshape([-1, self.brain.num_stacked_vector_observations])
                if self.use_visual_obs:
                    for i, _ in enumerate(self.model.visual_in):
                        _obs = np.array(buffer['visual_obs%d' % i][start:end])
                        if self.sequence_length > 1 and self.use_recurrent:
                            (_batch, _seq, _w, _h, _c) = _obs.shape
                            feed_dict[self.model.visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                        else:
                            feed_dict[self.model.visual_in[i]] = _obs
                    if self.use_curiosity:
                        for i, _ in enumerate(self.model.visual_in):
                            _obs = np.array(buffer['next_visual_obs%d' % i][start:end])
                            if self.sequence_length > 1 and self.use_recurrent:
                                (_batch, _seq, _w, _h, _c) = _obs.shape
                                feed_dict[self.model.next_visual_in[i]] = _obs.reshape([-1, _w, _h, _c])
                            else:
                                feed_dict[self.model.next_visual_in[i]] = _obs
                if self.use_recurrent:
                    mem_in = np.array(buffer['memory'][start:end])[:, 0, :]
                    feed_dict[self.model.memory_in] = mem_in

                run_list = [self.model.value_loss, self.model.policy_loss, self.model.update_batch]
                if self.use_curiosity:
                    run_list.extend([self.model.forward_loss, self.model.inverse_loss])
                values = self.sess.run(run_list, feed_dict=feed_dict)
                self.has_updated = True
                run_out = dict(zip(run_list, values))
                value_total.append(run_out[self.model.value_loss])
                policy_total.append(np.abs(run_out[self.model.policy_loss]))
                if self.use_curiosity:
                    inverse_total.append(run_out[self.model.inverse_loss])
                    forward_total.append(run_out[self.model.forward_loss])
        self.stats['value_loss'].append(np.mean(value_total))
        self.stats['policy_loss'].append(np.mean(policy_total))
        if self.use_curiosity:
            self.stats['forward_loss'].append(np.mean(forward_total))
            self.stats['inverse_loss'].append(np.mean(inverse_total))
        self.training_buffer.reset_update_buffer()


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

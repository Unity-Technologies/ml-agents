# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (PPO)
# Contains an implementation of PPO as described (https://arxiv.org/abs/1707.06347).

import logging
import os
from collections import deque

import numpy as np
import tensorflow as tf

from mlagents.envs import AllBrainInfo, BrainInfo
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.trainer import Trainer

logger = logging.getLogger("mlagents.trainers")


class PPOTrainer(Trainer):
    """The PPOTrainer is an implementation of the PPO algorithm."""

    def __init__(self, brain, reward_buff_cap, trainer_parameters, training, load, seed, run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The The identifier of the current run
        """
        super(PPOTrainer, self).__init__(brain, trainer_parameters, training, run_id)
        self.param_keys = ['batch_size', 'beta', 'buffer_size', 'epsilon', 'gamma', 'hidden_units', 'lambd',
                           'learning_rate', 'max_steps', 'normalize', 'num_epoch', 'num_layers',
                           'time_horizon', 'sequence_length', 'summary_freq', 'use_recurrent',
                           'summary_path', 'memory_size', 'use_curiosity', 'curiosity_strength',
                           'curiosity_enc_size', 'model_path']

        self.check_param_keys()
        self.use_curiosity = bool(trainer_parameters['use_curiosity'])
        self.step = 0
        self.policy = PPOPolicy(seed, brain, trainer_parameters,
                                self.is_training, load)

        stats = {'Environment/Cumulative Reward': [], 'Environment/Episode Length': [],
                 'Policy/Value Estimate': [], 'Policy/Entropy': [], 'Losses/Value Loss': [],
                 'Losses/Policy Loss': [], 'Policy/Learning Rate': []}
        if self.use_curiosity:
            stats['Losses/Forward Loss'] = []
            stats['Losses/Inverse Loss'] = []
            stats['Policy/Curiosity Reward'] = []
            self.intrinsic_rewards = {}
        self.stats = stats

        self.training_buffer = Buffer()
        self.cumulative_rewards = {}
        self._reward_buffer = deque(maxlen=reward_buff_cap)
        self.episode_steps = {}
        self.summary_path = trainer_parameters['summary_path']
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_writer = tf.summary.FileWriter(self.summary_path)

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
    def reward_buffer(self):
        """
        Returns the reward buffer. The reward buffer contains the cumulative
        rewards of the most recent episodes completed by agents using this
        trainer.
        :return: the reward buffer.
        """
        return self._reward_buffer

    def increment_step_and_update_last_reward(self):
        """
        Increment the step count of the trainer and Updates the last reward
        """
        if len(self.stats['Environment/Cumulative Reward']) > 0:
            mean_reward = np.mean(self.stats['Environment/Cumulative Reward'])
            self.policy.update_reward(mean_reward)
        self.policy.increment_step()
        self.step = self.policy.get_current_step()

    def take_action(self, all_brain_info: AllBrainInfo):
        """
        Decides actions given observations information, and takes them in environment.
        :param all_brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: a tuple containing action, memories, values and an object
        to be passed to add experiences
        """
        curr_brain_info = all_brain_info[self.brain_name]
        if len(curr_brain_info.agents) == 0:
            return [], [], [], None, None

        run_out = self.policy.evaluate(curr_brain_info)
        self.stats['Policy/Value Estimate'].append(run_out['value'].mean())
        self.stats['Policy/Entropy'].append(run_out['entropy'].mean())
        self.stats['Policy/Learning Rate'].append(run_out['learning_rate'])
        if self.policy.use_recurrent:
            return run_out['action'], run_out['memory_out'], None, \
                   run_out['value'], run_out
        else:
            return run_out['action'], None, None, run_out['value'], run_out

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
            if agent_brain_info is None:
                agent_brain_info = next_info
            agent_index = agent_brain_info.agents.index(agent_id)
            for i in range(len(next_info.visual_observations)):
                visual_observations[i].append(agent_brain_info.visual_observations[i][agent_index])
            vector_observations.append(agent_brain_info.vector_observations[agent_index])
            text_observations.append(agent_brain_info.text_observations[agent_index])
            if self.policy.use_recurrent:
                if len(agent_brain_info.memories > 0):
                    memories.append(agent_brain_info.memories[agent_index])
                else:
                    memories.append(self.policy.make_empty_memory(1))
            rewards.append(agent_brain_info.rewards[agent_index])
            local_dones.append(agent_brain_info.local_done[agent_index])
            max_reacheds.append(agent_brain_info.max_reached[agent_index])
            agents.append(agent_brain_info.agents[agent_index])
            prev_vector_actions.append(agent_brain_info.previous_vector_actions[agent_index])
            prev_text_actions.append(agent_brain_info.previous_text_actions[agent_index])
        if self.policy.use_recurrent:
            memories = np.vstack(memories)
        curr_info = BrainInfo(visual_observations, vector_observations, text_observations,
                              memories, rewards, agents, local_dones, prev_vector_actions,
                              prev_text_actions, max_reacheds)
        return curr_info

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

        if curr_info.agents != next_info.agents:
            curr_to_use = self.construct_curr_info(next_info)
        else:
            curr_to_use = curr_info

        intrinsic_rewards = self.policy.get_intrinsic_rewards(curr_to_use, next_info)

        for agent_id in next_info.agents:
            stored_info = self.training_buffer[agent_id].last_brain_info
            stored_take_action_outputs = self.training_buffer[agent_id].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        self.training_buffer[agent_id]['visual_obs%d' % i].append(
                            stored_info.visual_observations[i][idx])
                        self.training_buffer[agent_id]['next_visual_obs%d' % i].append(
                            next_info.visual_observations[i][next_idx])
                    if self.policy.use_vec_obs:
                        self.training_buffer[agent_id]['vector_obs'].append(stored_info.vector_observations[idx])
                        self.training_buffer[agent_id]['next_vector_in'].append(
                            next_info.vector_observations[next_idx])
                    if self.policy.use_recurrent:
                        if stored_info.memories.shape[1] == 0:
                            stored_info.memories = np.zeros((len(stored_info.agents), self.policy.m_size))
                        self.training_buffer[agent_id]['memory'].append(stored_info.memories[idx])
                    actions = stored_take_action_outputs['action']
                    if self.policy.use_continuous_act:
                        actions_pre = stored_take_action_outputs['pre_action']
                        self.training_buffer[agent_id]['actions_pre'].append(actions_pre[idx])
                        epsilons = stored_take_action_outputs['random_normal_epsilon']
                        self.training_buffer[agent_id]['random_normal_epsilon'].append(
                            epsilons[idx])
                    else:
                        self.training_buffer[agent_id]['action_mask'].append(
                            stored_info.action_masks[idx])
                    a_dist = stored_take_action_outputs['log_probs']
                    value = stored_take_action_outputs['value']
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
                    value_next = self.policy.get_value_estimate(bootstrapping_info, idx)

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
                                                          training_length=self.policy.sequence_length)

                self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
                    self.stats['Environment/Cumulative Reward'].append(
                        self.cumulative_rewards.get(agent_id, 0))
                    self.reward_buffer.appendleft(self.cumulative_rewards.get(agent_id, 0))
                    self.stats['Environment/Episode Length'].append(
                        self.episode_steps.get(agent_id, 0))
                    self.cumulative_rewards[agent_id] = 0
                    self.episode_steps[agent_id] = 0
                    if self.use_curiosity:
                        self.stats['Policy/Curiosity Reward'].append(
                            self.intrinsic_rewards.get(agent_id, 0))
                        self.intrinsic_rewards[agent_id] = 0

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset. 
        Get only called when the academy resets.
        """
        self.training_buffer.reset_local_buffers()
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
        return size_of_buffer > max(int(self.trainer_parameters['buffer_size'] / self.policy.sequence_length), 1)

    def update_policy(self):
        """
        Uses demonstration_buffer to update the policy.
        """
        n_sequences = max(int(self.trainer_parameters['batch_size'] / self.policy.sequence_length), 1)
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
                run_out = self.policy.update(buffer.make_mini_batch(start, end), n_sequences)
                value_total.append(run_out['value_loss'])
                policy_total.append(np.abs(run_out['policy_loss']))
                if self.use_curiosity:
                    inverse_total.append(run_out['inverse_loss'])
                    forward_total.append(run_out['forward_loss'])
        self.stats['Losses/Value Loss'].append(np.mean(value_total))
        self.stats['Losses/Policy Loss'].append(np.mean(policy_total))
        if self.use_curiosity:
            self.stats['Losses/Forward Loss'].append(np.mean(forward_total))
            self.stats['Losses/Inverse Loss'].append(np.mean(inverse_total))
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

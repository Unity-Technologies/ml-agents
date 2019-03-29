# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Behavioral Cloning)
# Contains an implementation of Behavioral Cloning Algorithm

import logging

import numpy as np
import tensorflow as tf

from mlagents.envs import AllBrainInfo
from mlagents.trainers.bc.policy import BCPolicy
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.trainer import Trainer

logger = logging.getLogger("mlagents.trainers")


class BCTrainer(Trainer):
    """The BCTrainer is an implementation of Behavioral Cloning."""

    def __init__(self, brain, trainer_parameters, training, load, seed,
                 run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The The identifier of the current run
        """
        super(BCTrainer, self).__init__(brain, trainer_parameters, training,
                                        run_id)
        self.policy = BCPolicy(seed, brain, trainer_parameters, load)
        self.n_sequences = 1
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.stats = {'Losses/Cloning Loss': [], 'Environment/Episode Length': [],
                      'Environment/Cumulative Reward': []}

        self.batches_per_epoch = trainer_parameters['batches_per_epoch']


        self.demonstration_buffer = Buffer()
        self.evaluation_buffer = Buffer()

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
        return self.policy.get_current_step()

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        if len(self.stats['Environment/Cumulative Reward']) > 0:
            return np.mean(self.stats['Environment/Cumulative Reward'])
        else:
            return 0

    def increment_step_and_update_last_reward(self):
        """
        Increment the step count of the trainer and Updates the last reward
        """
        self.policy.increment_step()
        return

    def add_experiences(self, curr_info: AllBrainInfo, next_info: AllBrainInfo,
                        take_action_outputs):
        """
        Adds experiences to each agent's experience history.
        :param curr_info: Current AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param next_info: Next AllBrainInfo (Dictionary of all current brains and corresponding BrainInfo).
        :param take_action_outputs: The outputs of the take action method.
        """

        # Used to collect information about student performance.
        info_student = curr_info[self.brain_name]
        next_info_student = next_info[self.brain_name]
        for agent_id in info_student.agents:
            self.evaluation_buffer[agent_id].last_brain_info = info_student

        for agent_id in next_info_student.agents:
            stored_info_student = self.evaluation_buffer[agent_id].last_brain_info
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
        info_student = next_info[self.brain_name]
        for l in range(len(info_student.agents)):
            if info_student.local_done[l]:
                agent_id = info_student.agents[l]
                self.stats['Environment/Cumulative Reward'].append(
                    self.cumulative_rewards.get(agent_id, 0))
                self.stats['Environment/Episode Length'].append(
                    self.episode_steps.get(agent_id, 0))
                self.cumulative_rewards[agent_id] = 0
                self.episode_steps[agent_id] = 0

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        self.evaluation_buffer.reset_local_buffers()
        for agent_id in self.cumulative_rewards:
            self.cumulative_rewards[agent_id] = 0
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        return len(self.demonstration_buffer.update_buffer['actions']) > self.n_sequences

    def update_policy(self):
        """
        Updates the policy.
        """
        self.demonstration_buffer.update_buffer.shuffle()
        batch_losses = []
        num_batches = min(len(self.demonstration_buffer.update_buffer['actions']) //
                          self.n_sequences, self.batches_per_epoch)
        for i in range(num_batches):
            update_buffer = self.demonstration_buffer.update_buffer
            start = i * self.n_sequences
            end = (i + 1) * self.n_sequences
            mini_batch = update_buffer.make_mini_batch(start, end)
            run_out = self.policy.update(mini_batch, self.n_sequences)
            loss = run_out['policy_loss']
            batch_losses.append(loss)
        if len(batch_losses) > 0:
            self.stats['Losses/Cloning Loss'].append(np.mean(batch_losses))
        else:
            self.stats['Losses/Cloning Loss'].append(0)

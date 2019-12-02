# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Behavioral Cloning)
# Contains an implementation of Behavioral Cloning Algorithm

import logging

import numpy as np

from mlagents.envs.brain import BrainInfo
from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.trainers.bc.policy import BCPolicy
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.trainer import Trainer

logger = logging.getLogger("mlagents.trainers")


class BCTrainer(Trainer):
    """The BCTrainer is an implementation of Behavioral Cloning."""

    def __init__(self, brain, trainer_parameters, training, load, seed, run_id):
        """
        Responsible for collecting experiences and training PPO model.
        :param  trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The identifier of the current run
        """
        super(BCTrainer, self).__init__(brain, trainer_parameters, training, run_id)
        self.policy = BCPolicy(seed, brain, trainer_parameters, load)
        self.n_sequences = 1
        self.cumulative_rewards = {}
        self.episode_steps = {}
        self.stats = {
            "Losses/Cloning Loss": [],
            "Environment/Episode Length": [],
            "Environment/Cumulative Reward": [],
        }

        self.batches_per_epoch = trainer_parameters["batches_per_epoch"]

        self.demonstration_buffer = Buffer()
        self.evaluation_buffer = Buffer()

    def add_experiences(
        self,
        curr_info: BrainInfo,
        next_info: BrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_info: Current BrainInfo
        :param next_info: Next BrainInfo
        :param take_action_outputs: The outputs of the take action method.
        """

        # Used to collect information about student performance.
        for agent_id in curr_info.agents:
            self.evaluation_buffer[agent_id].last_brain_info = curr_info

        for agent_id in next_info.agents:
            stored_next_info = self.evaluation_buffer[agent_id].last_brain_info
            if stored_next_info is None:
                continue
            else:
                next_idx = next_info.agents.index(agent_id)
                if agent_id not in self.cumulative_rewards:
                    self.cumulative_rewards[agent_id] = 0
                self.cumulative_rewards[agent_id] += next_info.rewards[next_idx]
                if not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1

    def process_experiences(
        self, current_info: BrainInfo, next_info: BrainInfo
    ) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo
        :param next_info: Next BrainInfo
        """
        for l in range(len(next_info.agents)):
            if next_info.local_done[l]:
                agent_id = next_info.agents[l]
                self.stats["Environment/Cumulative Reward"].append(
                    self.cumulative_rewards.get(agent_id, 0)
                )
                self.stats["Environment/Episode Length"].append(
                    self.episode_steps.get(agent_id, 0)
                )
                self.reward_buffer.appendleft(self.cumulative_rewards.get(agent_id, 0))
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
        return (
            len(self.demonstration_buffer.update_buffer["actions"]) > self.n_sequences
        )

    def update_policy(self):
        """
        Updates the policy.
        """
        self.demonstration_buffer.update_buffer.shuffle(self.policy.sequence_length)
        batch_losses = []
        batch_size = self.n_sequences * self.policy.sequence_length
        # We either divide the entire buffer into num_batches batches, or limit the number
        # of batches to batches_per_epoch.
        num_batches = min(
            len(self.demonstration_buffer.update_buffer["actions"]) // batch_size,
            self.batches_per_epoch,
        )

        for i in range(0, num_batches * batch_size, batch_size):
            update_buffer = self.demonstration_buffer.update_buffer
            mini_batch = update_buffer.make_mini_batch(i, i + batch_size)
            run_out = self.policy.update(mini_batch, self.n_sequences)
            loss = run_out["policy_loss"]
            batch_losses.append(loss)
        if len(batch_losses) > 0:
            self.stats["Losses/Cloning Loss"].append(np.mean(batch_losses))
        else:
            self.stats["Losses/Cloning Loss"].append(0)

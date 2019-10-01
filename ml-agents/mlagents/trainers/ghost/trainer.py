# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (Ghost)

import logging
import random

import numpy as np
import tensorflow as tf
from typing import List

from mlagents.envs import AllBrainInfo, BrainInfo
from mlagents.trainers.ppo.policy import PPOPolicy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.trainer import UnityTrainerException
from mlagents.trainers import ActionInfo, combine

logger = logging.getLogger("mlagents.trainers")


class GhostTrainer(PPOTrainer):

    def __init__(self, brain, reward_buff_cap, trainer_parameters, seed, run_id):
        """
        Responsible for loading models saved by his master trainer and assigning different policies to its agents .
        :param trainer_parameters: The parameters for the trainer (dictionary). Those are based on his master trainer parameters.
        :param seed: The seed the model will be initialized with
        :param run_id: The The identifier of the current run
        """
        super(GhostTrainer, self).__init__(brain, reward_buff_cap,
                                           trainer_parameters, False, False, seed, run_id)
        self.agent_policies = {}
        self.ghost_recent_ckpts_threshold = trainer_parameters['ghost_recent_ckpts_threshold']
        self.ghost_prob_sample_only_recent = trainer_parameters['ghost_prob_sample_only_recent']
        self.policies = []
        for _ in range(trainer_parameters['ghost_num_policies']):
            self.policies.append(
                PPOPolicy(
                    seed,
                    brain,
                    trainer_parameters,
                    self.is_training,
                    False
                )
            )

    @staticmethod
    def should_update_elo_rating():
        """
        Should the elo rating be updated
        :return: True if not ghost else False
        """
        return False

    
    def add_experiences(
        self,
        curr_all_info: AllBrainInfo,
        next_all_info: AllBrainInfo,
        take_action_outputs,
    ):
        """
        No need to collect experiences since we don't update the ghost brain
        """
        pass

    def process_experiences(self, current_info: AllBrainInfo, new_info: AllBrainInfo):
        """
        No need to collect experiences since we don't update the ghost brain
        """
        pass

    def get_elo_rating(self):
        return np.mean([policy.get_elo_rating() for policy in self.agent_policies.values()])

    def get_action(self, curr_info: BrainInfo) -> ActionInfo:
        """
        Get an action using this trainer's current policy.
        :param curr_info: Current BrainInfo.
        :return: The ActionInfo given by the policy given the BrainInfo.
        """
        self.trainer_metrics.start_experience_collection_timer()
        if len(curr_info.agents) == 0:
            return self.policy.get_action(curr_info)

        action_infos: List[ActionInfo] = [None] * len(curr_info.agents)
        for idx, agent in enumerate(curr_info.agent_ids):
            if agent not in self.agent_policies:
                self.agent_policies[agent] = random.choice(self.policies)

            agent_brain_info = curr_info.get_agent_brain_info(agent)
            action_infos[idx] = self.agent_policies[agent].get_action(agent_brain_info)

        result = combine(action_infos)
        self.trainer_metrics.end_experience_collection_timer()
        return result

    def is_ready_update(self):
        """
        Ghost trainers are never updating
        """
        return False

    def update_policy(self):
        """
        Ghost trainers are never updating
        """
        raise UnityTrainerException("The ghost trainer's policy shouldn't get updated.")

    def save_model(self):
        """
        Don't save the model for ghost trainers
        """
        pass

    def end_episode(self):
        """
        Get called when the academy resets.
        Loads a random saved model for each policy
        """
        # Model path is the same as the master trainer's model path
        ckpt_state = tf.train.get_checkpoint_state(self.policy.model_path)
        if ckpt_state:
            all_ckpts = tf.train.get_checkpoint_state(self.policy.model_path).all_model_checkpoint_paths

            if all_ckpts:
                for policy in self.policies:
                    policy_used_ckpts = all_ckpts

                    # There is a 1-p_load_from_all_checkpoints probability that we sample the policy only from the last load_from_last_N_checkpoints policies
                    if random.random() > self.ghost_prob_sample_only_recent:
                        policy_used_ckpts = policy_used_ckpts[-self.ghost_recent_ckpts_threshold:]

                    checkpoint_path = random.sample(policy_used_ckpts, 1)[0]
                    policy.load_graph(checkpoint_path)

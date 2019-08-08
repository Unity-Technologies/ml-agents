from typing import Any, Dict, List
import numpy as np
from mlagents.envs.brain import BrainInfo

import tensorflow as tf

from mlagents.trainers.buffer import Buffer
from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.components.reward_signals.curiosity.model import CuriosityModel
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.models import LearningModel


class CuriosityRewardSignal(RewardSignal):
    def __init__(
        self,
        policy: TFPolicy,
        policy_model: LearningModel,
        strength: float,
        gamma: float,
        encoding_size: int = 128,
        learning_rate: float = 3e-4,
    ):
        """
        Creates the Curiosity reward generator
        :param policy: The Learning Policy
        :param strength: The scaling parameter for the reward. The scaled reward will be the unscaled
        reward multiplied by the strength parameter
        :param gamma: The time discounting factor used for this reward.
        :param encoding_size: The size of the hidden encoding layer for the ICM
        :param learning_rate: The learning rate for the ICM.
        """
        super().__init__(policy, policy_model, strength, gamma)
        self.model = CuriosityModel(
            policy_model, encoding_size=encoding_size, learning_rate=learning_rate
        )
        self.use_terminal_states = False
        self.update_dict = {
            "curiosity_forward_loss": self.model.forward_loss,
            "curiosity_inverse_loss": self.model.inverse_loss,
            "curiosity_update": self.model.update_batch,
        }
        self.stats_name_to_update_name = {
            "Losses/Curiosity Forward Loss": "curiosity_forward_loss",
            "Losses/Curiosity Inverse Loss": "curiosity_inverse_loss",
        }
        self.has_updated = False

    def evaluate(
        self, current_info: BrainInfo, next_info: BrainInfo
    ) -> RewardSignalResult:
        """
        Evaluates the reward for the agents present in current_info given the next_info
        :param current_info: The current BrainInfo.
        :param next_info: The BrainInfo from the next timestep.
        :return: a RewardSignalResult of (scaled intrinsic reward, unscaled intrinsic reward) provided by the generator
        """
        if len(current_info.agents) == 0:
            return []

        feed_dict = {
            self.policy.model.batch_size: len(next_info.vector_observations),
            self.policy.model.sequence_length: 1,
        }
        feed_dict = self.policy.fill_eval_dict(feed_dict, brain_info=current_info)
        if self.policy.use_continuous_act:
            feed_dict[
                self.policy.model.selected_actions
            ] = next_info.previous_vector_actions
        else:
            feed_dict[
                self.policy.model.action_holder
            ] = next_info.previous_vector_actions
        for i in range(self.policy.model.vis_obs_size):
            feed_dict[self.model.next_visual_in[i]] = next_info.visual_observations[i]
        if self.policy.use_vec_obs:
            feed_dict[self.model.next_vector_in] = next_info.vector_observations
        unscaled_reward = self.policy.sess.run(
            self.model.intrinsic_reward, feed_dict=feed_dict
        )
        scaled_reward = np.clip(
            unscaled_reward * float(self.has_updated) * self.strength, 0, 1
        )
        return RewardSignalResult(scaled_reward, unscaled_reward)

    @classmethod
    def check_config(
        cls, config_dict: Dict[str, Any], param_keys: List[str] = None
    ) -> None:
        """
        Checks the config and throw an exception if a hyperparameter is missing. Curiosity requires strength,
        gamma, and encoding size at minimum.
        """
        param_keys = ["strength", "gamma", "encoding_size"]
        super().check_config(config_dict, param_keys)

    def prepare_update(
        self,
        policy_model: LearningModel,
        mini_batch: Dict[str, np.ndarray],
        num_sequences: int,
    ) -> Dict[tf.Tensor, Any]:
        """
        Prepare for update and get feed_dict.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Feed_dict needed for update.
        """
        feed_dict = {
            policy_model.batch_size: num_sequences,
            policy_model.sequence_length: self.policy.sequence_length,
            policy_model.mask_input: mini_batch["masks"],
            policy_model.advantage: mini_batch["advantages"],
            policy_model.all_old_log_probs: mini_batch["action_probs"],
        }
        if self.policy.use_continuous_act:
            feed_dict[policy_model.output_pre] = mini_batch["actions_pre"]
        else:
            feed_dict[policy_model.action_holder] = mini_batch["actions"]
        if self.policy.use_vec_obs:
            feed_dict[policy_model.vector_in] = mini_batch["vector_obs"]
            feed_dict[self.model.next_vector_in] = mini_batch["next_vector_in"]
        if policy_model.vis_obs_size > 0:
            for i, _ in enumerate(policy_model.visual_in):
                feed_dict[policy_model.visual_in[i]] = mini_batch["visual_obs%d" % i]
            for i, _ in enumerate(policy_model.visual_in):
                feed_dict[self.model.next_visual_in[i]] = mini_batch[
                    "next_visual_obs%d" % i
                ]

        self.has_updated = True
        return feed_dict

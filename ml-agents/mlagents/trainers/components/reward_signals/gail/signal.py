from typing import Any, Dict, List
import logging
import numpy as np
import tensorflow as tf

from mlagents.envs.brain import BrainInfo
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.models import LearningModel
from .model import GAILModel
from mlagents.trainers.demo_loader import demo_to_buffer

LOGGER = logging.getLogger("mlagents.trainers")


class GAILRewardSignal(RewardSignal):
    def __init__(
        self,
        policy: TFPolicy,
        policy_model: LearningModel,
        strength: float,
        gamma: float,
        demo_path: str,
        encoding_size: int = 64,
        learning_rate: float = 3e-4,
        use_actions: bool = False,
        use_vail: bool = False,
    ):
        """
        The GAIL Reward signal generator. https://arxiv.org/abs/1606.03476
        :param policy: The policy of the learning model
        :param strength: The scaling parameter for the reward. The scaled reward will be the unscaled
        reward multiplied by the strength parameter
        :param gamma: The time discounting factor used for this reward.
        :param demo_path: The path to the demonstration file
        :param num_epoch: The number of epochs to train over the training buffer for the discriminator.
        :param encoding_size: The size of the the hidden layers of the discriminator
        :param learning_rate: The Learning Rate used during GAIL updates.
        :param use_actions: Whether or not to use the actions for the discriminator.
        :param use_vail: Whether or not to use a variational bottleneck for the discriminator.
        See https://arxiv.org/abs/1810.00821.
        """
        super().__init__(policy, policy_model, strength, gamma)
        self.use_terminal_states = False

        self.model = GAILModel(
            policy.model, 128, learning_rate, encoding_size, use_actions, use_vail
        )
        _, self.demonstration_buffer = demo_to_buffer(demo_path, policy.sequence_length)
        self.has_updated = False
        self.update_dict: Dict[str, tf.Tensor] = {
            "gail_loss": self.model.loss,
            "gail_update_batch": self.model.update_batch,
            "gail_policy_estimate": self.model.policy_estimate,
            "gail_expert_estimate": self.model.expert_estimate,
        }
        if self.model.use_vail:
            self.update_dict["kl_loss"] = self.model.kl_loss
            self.update_dict["z_log_sigma_sq"] = self.model.z_log_sigma_sq
            self.update_dict["z_mean_expert"] = self.model.z_mean_expert
            self.update_dict["z_mean_policy"] = self.model.z_mean_policy
            self.update_dict["beta_update"] = self.model.update_beta

        self.stats_name_to_update_name = {"Losses/GAIL Loss": "gail_loss"}

    def evaluate(
        self, current_info: BrainInfo, next_info: BrainInfo
    ) -> RewardSignalResult:
        if len(current_info.agents) == 0:
            return []

        feed_dict: Dict[tf.Tensor, Any] = {
            self.policy.model.batch_size: len(next_info.vector_observations),
            self.policy.model.sequence_length: 1,
        }
        if self.model.use_vail:
            feed_dict[self.model.use_noise] = [0]

        feed_dict = self.policy.fill_eval_dict(feed_dict, brain_info=current_info)
        feed_dict[self.model.done_policy] = np.reshape(next_info.local_done, [-1, 1])
        if self.policy.use_continuous_act:
            feed_dict[
                self.policy.model.selected_actions
            ] = next_info.previous_vector_actions
        else:
            feed_dict[
                self.policy.model.action_holder
            ] = next_info.previous_vector_actions
        unscaled_reward = self.policy.sess.run(
            self.model.intrinsic_reward, feed_dict=feed_dict
        )
        scaled_reward = unscaled_reward * float(self.has_updated) * self.strength
        return RewardSignalResult(scaled_reward, unscaled_reward)

    @classmethod
    def check_config(
        cls, config_dict: Dict[str, Any], param_keys: List[str] = None
    ) -> None:
        """
        Checks the config and throw an exception if a hyperparameter is missing. GAIL requires strength and gamma
        at minimum.
        """
        param_keys = ["strength", "gamma", "demo_path"]
        super().check_config(config_dict, param_keys)

    def prepare_update(
        self,
        policy_model: LearningModel,
        mini_batch_policy: Dict[str, np.ndarray],
        num_sequences: int,
    ) -> Dict[tf.Tensor, Any]:
        """
        Prepare inputs for update. .
        :param mini_batch_demo: A mini batch of expert trajectories
        :param mini_batch_policy: A mini batch of trajectories sampled from the current policy
        :return: Feed_dict for update process.
        """
        max_num_experiences = min(
            len(mini_batch_policy["actions"]),
            len(self.demonstration_buffer.update_buffer["actions"]),
        )
        # If num_sequences is less, we need to shorten the input batch.
        for key, element in mini_batch_policy.items():
            mini_batch_policy[key] = element[:max_num_experiences]
        # Get demo buffer
        self.demonstration_buffer.update_buffer.shuffle(1)
        # TODO: Replace with SAC sample method
        mini_batch_demo = self.demonstration_buffer.update_buffer.make_mini_batch(
            0, len(mini_batch_policy["actions"])
        )

        feed_dict: Dict[tf.Tensor, Any] = {
            self.model.done_expert_holder: mini_batch_demo["done"],
            self.model.done_policy_holder: mini_batch_policy["done"],
        }

        if self.model.use_vail:
            feed_dict[self.model.use_noise] = [1]

        feed_dict[self.model.action_in_expert] = np.array(mini_batch_demo["actions"])
        if self.policy.use_continuous_act:
            feed_dict[policy_model.selected_actions] = mini_batch_policy["actions"]
        else:
            feed_dict[policy_model.action_holder] = mini_batch_policy["actions"]

        if self.policy.use_vis_obs > 0:
            for i in range(len(policy_model.visual_in)):
                feed_dict[policy_model.visual_in[i]] = mini_batch_policy[
                    "visual_obs%d" % i
                ]
                feed_dict[self.model.expert_visual_in[i]] = mini_batch_demo[
                    "visual_obs%d" % i
                ]
        if self.policy.use_vec_obs:
            feed_dict[policy_model.vector_in] = mini_batch_policy["vector_obs"]
            feed_dict[self.model.obs_in_expert] = mini_batch_demo["vector_obs"]
        self.has_updated = True
        return feed_dict

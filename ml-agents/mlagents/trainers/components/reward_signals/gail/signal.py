from typing import Any, Dict, List
import numpy as np
from mlagents.tf_utils import tf

from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.policy.tf_policy import TFPolicy
from .model import GAILModel
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.buffer import AgentBuffer


class GAILRewardSignal(RewardSignal):
    def __init__(
        self,
        policy: TFPolicy,
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
        super().__init__(policy, strength, gamma)
        self.use_terminal_states = False

        self.model = GAILModel(
            policy, 128, learning_rate, encoding_size, use_actions, use_vail
        )
        _, self.demonstration_buffer = demo_to_buffer(demo_path, policy.sequence_length)
        self.has_updated = False
        self.update_dict: Dict[str, tf.Tensor] = {
            "gail_loss": self.model.loss,
            "gail_update_batch": self.model.update_batch,
            "gail_policy_estimate": self.model.mean_policy_estimate,
            "gail_expert_estimate": self.model.mean_expert_estimate,
        }
        if self.model.use_vail:
            self.update_dict["kl_loss"] = self.model.kl_loss
            self.update_dict["z_log_sigma_sq"] = self.model.z_log_sigma_sq
            self.update_dict["z_mean_expert"] = self.model.z_mean_expert
            self.update_dict["z_mean_policy"] = self.model.z_mean_policy
            self.update_dict["beta_update"] = self.model.update_beta

        self.stats_name_to_update_name = {
            "Losses/GAIL Loss": "gail_loss",
            "Policy/GAIL Policy Estimate": "gail_policy_estimate",
            "Policy/GAIL Expert Estimate": "gail_expert_estimate",
        }

    def evaluate_batch(self, mini_batch: AgentBuffer) -> RewardSignalResult:
        feed_dict: Dict[tf.Tensor, Any] = {
            self.policy.batch_size_ph: len(mini_batch["actions"]),
            self.policy.sequence_length_ph: self.policy.sequence_length,
        }
        if self.model.use_vail:
            feed_dict[self.model.use_noise] = [0]

        if self.policy.use_vec_obs:
            feed_dict[self.policy.vector_in] = mini_batch["vector_obs"]
        if self.policy.vis_obs_size > 0:
            for i in range(len(self.policy.visual_in)):
                _obs = mini_batch["visual_obs%d" % i]
                feed_dict[self.policy.visual_in[i]] = _obs

        if self.policy.use_continuous_act:
            feed_dict[self.policy.selected_actions] = mini_batch["actions"]
        else:
            feed_dict[self.policy.output] = mini_batch["actions"]
        feed_dict[self.model.done_policy_holder] = np.array(
            mini_batch["done"]
        ).flatten()
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
        self, policy: TFPolicy, mini_batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        """
        Prepare inputs for update. .
        :param mini_batch_demo: A mini batch of expert trajectories
        :param mini_batch_policy: A mini batch of trajectories sampled from the current policy
        :return: Feed_dict for update process.
        """
        # Get batch from demo buffer. Even if demo buffer is smaller, we sample with replacement
        mini_batch_demo = self.demonstration_buffer.sample_mini_batch(
            mini_batch.num_experiences, 1
        )

        feed_dict: Dict[tf.Tensor, Any] = {
            self.model.done_expert_holder: mini_batch_demo["done"],
            self.model.done_policy_holder: mini_batch["done"],
        }

        if self.model.use_vail:
            feed_dict[self.model.use_noise] = [1]

        feed_dict[self.model.action_in_expert] = np.array(mini_batch_demo["actions"])
        if self.policy.use_continuous_act:
            feed_dict[policy.selected_actions] = mini_batch["actions"]
        else:
            feed_dict[policy.output] = mini_batch["actions"]

        if self.policy.use_vis_obs > 0:
            for i in range(len(policy.visual_in)):
                feed_dict[policy.visual_in[i]] = mini_batch["visual_obs%d" % i]
                feed_dict[self.model.expert_visual_in[i]] = mini_batch_demo[
                    "visual_obs%d" % i
                ]
        if self.policy.use_vec_obs:
            feed_dict[policy.vector_in] = mini_batch["vector_obs"]
            feed_dict[self.model.obs_in_expert] = mini_batch_demo["vector_obs"]
        self.has_updated = True
        return feed_dict

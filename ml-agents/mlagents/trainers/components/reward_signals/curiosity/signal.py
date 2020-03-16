from typing import Any, Dict, List
import numpy as np
from mlagents.tf_utils import tf

from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.components.reward_signals.curiosity.model import CuriosityModel
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.buffer import AgentBuffer


class CuriosityRewardSignal(RewardSignal):
    def __init__(
        self,
        policy: TFPolicy,
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
        super().__init__(policy, strength, gamma)
        self.model = CuriosityModel(
            policy, encoding_size=encoding_size, learning_rate=learning_rate
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

    def evaluate_batch(self, mini_batch: AgentBuffer) -> RewardSignalResult:
        feed_dict: Dict[tf.Tensor, Any] = {
            self.policy.batch_size_ph: len(mini_batch["actions"]),
            self.policy.sequence_length_ph: self.policy.sequence_length,
        }
        if self.policy.use_vec_obs:
            feed_dict[self.policy.vector_in] = mini_batch["vector_obs"]
            feed_dict[self.model.next_vector_in] = mini_batch["next_vector_in"]
        if self.policy.vis_obs_size > 0:
            for i in range(len(self.policy.visual_in)):
                _obs = mini_batch["visual_obs%d" % i]
                _next_obs = mini_batch["next_visual_obs%d" % i]
                feed_dict[self.policy.visual_in[i]] = _obs
                feed_dict[self.model.next_visual_in[i]] = _next_obs

        if self.policy.use_continuous_act:
            feed_dict[self.policy.selected_actions] = mini_batch["actions"]
        else:
            feed_dict[self.policy.output] = mini_batch["actions"]
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
        self, policy: TFPolicy, mini_batch: AgentBuffer, num_sequences: int
    ) -> Dict[tf.Tensor, Any]:
        """
        Prepare for update and get feed_dict.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Feed_dict needed for update.
        """
        feed_dict = {
            policy.batch_size_ph: num_sequences,
            policy.sequence_length_ph: self.policy.sequence_length,
            policy.mask_input: mini_batch["masks"],
        }
        if self.policy.use_continuous_act:
            feed_dict[policy.selected_actions] = mini_batch["actions"]
        else:
            feed_dict[policy.output] = mini_batch["actions"]
        if self.policy.use_vec_obs:
            feed_dict[policy.vector_in] = mini_batch["vector_obs"]
            feed_dict[self.model.next_vector_in] = mini_batch["next_vector_in"]
        if policy.vis_obs_size > 0:
            for i, vis_in in enumerate(policy.visual_in):
                feed_dict[vis_in] = mini_batch["visual_obs%d" % i]
            for i, next_vis_in in enumerate(self.model.next_visual_in):
                feed_dict[next_vis_in] = mini_batch["next_visual_obs%d" % i]

        self.has_updated = True
        return feed_dict

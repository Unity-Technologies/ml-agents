from typing import Any, Dict
import numpy as np
from mlagents.tf_utils import tf

from mlagents.trainers.tf.components.reward_signals import (
    RewardSignal,
    RewardSignalResult,
)
from mlagents.trainers.tf.components.reward_signals.curiosity.model import (
    CuriosityModel,
)
from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.settings import CuriositySettings


class CuriosityRewardSignal(RewardSignal):
    def __init__(self, policy: TFPolicy, settings: CuriositySettings):
        """
        Creates the Curiosity reward generator
        :param policy: The Learning Policy
        :param settings: CuriositySettings object that contains the parameters
            (including encoding size and learning rate) for this CuriosityRewardSignal.
        """
        super().__init__(policy, settings)
        self.model = CuriosityModel(
            policy,
            encoding_size=settings.encoding_size,
            learning_rate=settings.learning_rate,
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
            self.policy.batch_size_ph: len(mini_batch["vector_obs"]),
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
            feed_dict[self.policy.selected_actions] = mini_batch["continuous_action"]
        else:
            feed_dict[self.policy.output] = mini_batch["discrete_action"]
        unscaled_reward = self.policy.sess.run(
            self.model.intrinsic_reward, feed_dict=feed_dict
        )
        scaled_reward = np.clip(
            unscaled_reward * float(self.has_updated) * self.strength, 0, 1
        )
        return RewardSignalResult(scaled_reward, unscaled_reward)

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
            feed_dict[policy.selected_actions] = mini_batch["continuous_action"]
        else:
            feed_dict[policy.output] = mini_batch["discrete_action"]
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

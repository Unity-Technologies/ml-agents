from typing import Any, Dict, List
import logging
import numpy as np
import tensorflow as tf

from mlagents.envs.brain import BrainInfo
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.components.reward_signals import RewardSignal, RewardSignalResult
from mlagents.trainers.tf_policy import TFPolicy
from .model import GAILModel
from mlagents.trainers.demo_loader import demo_to_buffer

LOGGER = logging.getLogger("mlagents.trainers")


class GAILRewardSignal(RewardSignal):
    def __init__(
        self,
        policy: TFPolicy,
        strength: float,
        gamma: float,
        demo_path: str,
        num_epoch: int = 3,
        encoding_size: int = 64,
        learning_rate: float = 3e-4,
        samples_per_update: int = 0,
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
        :param samples_per_update: The maximum number of samples to update during GAIL updates.
        :param use_actions: Whether or not to use the actions for the discriminator.
        :param use_vail: Whether or not to use a variational bottleneck for the discriminator.
        See https://arxiv.org/abs/1810.00821.
        """
        super().__init__(policy, strength, gamma)
        self.num_epoch = num_epoch
        self.samples_per_update = samples_per_update
        self.use_terminal_states = False

        self.model = GAILModel(
            policy.model, 128, learning_rate, encoding_size, use_actions, use_vail
        )
        _, self.demonstration_buffer = demo_to_buffer(demo_path, policy.sequence_length)
        self.has_updated = False

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
        if self.policy.use_recurrent:
            if current_info.memories.shape[1] == 0:
                current_info.memories = self.policy.make_empty_memory(
                    len(current_info.agents)
                )
            feed_dict[self.policy.model.memory_in] = current_info.memories
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

    def update(self, update_buffer: Buffer, n_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param update_buffer: The policy buffer containing the trajectories for the current policy.
        :param n_sequences: The number of sequences from demo and policy used in each mini batch.
        :return: The loss of the update.
        """
        batch_losses = []
        # Divide by 2 since we have two buffers, so we have roughly the same batch size
        n_sequences = max(n_sequences // 2, 1)
        possible_demo_batches = (
            len(self.demonstration_buffer.update_buffer["actions"]) // n_sequences
        )
        possible_policy_batches = len(update_buffer["actions"]) // n_sequences
        possible_batches = min(possible_policy_batches, possible_demo_batches)

        max_batches = self.samples_per_update // n_sequences

        kl_loss = []
        policy_estimate = []
        expert_estimate = []
        z_log_sigma_sq = []
        z_mean_expert = []
        z_mean_policy = []

        n_epoch = self.num_epoch
        for _epoch in range(n_epoch):
            self.demonstration_buffer.update_buffer.shuffle()
            update_buffer.shuffle()
            if max_batches == 0:
                num_batches = possible_batches
            else:
                num_batches = min(possible_batches, max_batches)
            for i in range(num_batches):
                demo_update_buffer = self.demonstration_buffer.update_buffer
                policy_update_buffer = update_buffer
                start = i * n_sequences
                end = (i + 1) * n_sequences
                mini_batch_demo = demo_update_buffer.make_mini_batch(start, end)
                mini_batch_policy = policy_update_buffer.make_mini_batch(start, end)
                run_out = self._update_batch(mini_batch_demo, mini_batch_policy)
                loss = run_out["gail_loss"]

                policy_estimate.append(run_out["policy_estimate"])
                expert_estimate.append(run_out["expert_estimate"])
                if self.model.use_vail:
                    kl_loss.append(run_out["kl_loss"])
                    z_log_sigma_sq.append(run_out["z_log_sigma_sq"])
                    z_mean_policy.append(run_out["z_mean_policy"])
                    z_mean_expert.append(run_out["z_mean_expert"])

                batch_losses.append(loss)
        self.has_updated = True

        print_list = ["n_epoch", "beta", "policy_estimate", "expert_estimate"]
        print_vals = [
            n_epoch,
            self.policy.sess.run(self.model.beta),
            np.mean(policy_estimate),
            np.mean(expert_estimate),
        ]
        if self.model.use_vail:
            print_list += [
                "kl_loss",
                "z_mean_expert",
                "z_mean_policy",
                "z_log_sigma_sq",
            ]
            print_vals += [
                np.mean(kl_loss),
                np.mean(z_mean_expert),
                np.mean(z_mean_policy),
                np.mean(z_log_sigma_sq),
            ]
        LOGGER.debug(
            "GAIL Debug:\n\t\t"
            + "\n\t\t".join(
                "{0}: {1}".format(_name, _val)
                for _name, _val in zip(print_list, print_vals)
            )
        )
        update_stats = {"Losses/GAIL Loss": np.mean(batch_losses)}
        return update_stats

    def _update_batch(
        self,
        mini_batch_demo: Dict[str, np.ndarray],
        mini_batch_policy: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Helper method for update.
        :param mini_batch_demo: A mini batch of expert trajectories
        :param mini_batch_policy: A mini batch of trajectories sampled from the current policy
        :return: Output from update process.
        """
        feed_dict: Dict[tf.Tensor, Any] = {
            self.model.done_expert: mini_batch_demo["done"].reshape([-1, 1]),
            self.model.done_policy: mini_batch_policy["done"].reshape([-1, 1]),
        }

        if self.model.use_vail:
            feed_dict[self.model.use_noise] = [1]

        if self.policy.use_continuous_act:
            feed_dict[self.policy.model.selected_actions] = mini_batch_policy[
                "actions"
            ].reshape([-1, self.policy.model.act_size[0]])
            feed_dict[self.model.action_in_expert] = mini_batch_demo["actions"].reshape(
                [-1, self.policy.model.act_size[0]]
            )
        else:
            feed_dict[self.policy.model.action_holder] = mini_batch_policy[
                "actions"
            ].reshape([-1, len(self.policy.model.act_size)])
            feed_dict[self.model.action_in_expert] = mini_batch_demo["actions"].reshape(
                [-1, len(self.policy.model.act_size)]
            )

        if self.policy.use_vis_obs > 0:
            for i in range(len(self.policy.model.visual_in)):
                policy_obs = mini_batch_policy["visual_obs%d" % i]
                if self.policy.sequence_length > 1 and self.policy.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = policy_obs.shape
                    feed_dict[self.policy.model.visual_in[i]] = policy_obs.reshape(
                        [-1, _w, _h, _c]
                    )
                else:
                    feed_dict[self.policy.model.visual_in[i]] = policy_obs

                demo_obs = mini_batch_demo["visual_obs%d" % i]
                if self.policy.sequence_length > 1 and self.policy.use_recurrent:
                    (_batch, _seq, _w, _h, _c) = demo_obs.shape
                    feed_dict[self.model.expert_visual_in[i]] = demo_obs.reshape(
                        [-1, _w, _h, _c]
                    )
                else:
                    feed_dict[self.model.expert_visual_in[i]] = demo_obs
        if self.policy.use_vec_obs:
            feed_dict[self.policy.model.vector_in] = mini_batch_policy[
                "vector_obs"
            ].reshape([-1, self.policy.vec_obs_size])
            feed_dict[self.model.obs_in_expert] = mini_batch_demo["vector_obs"].reshape(
                [-1, self.policy.vec_obs_size]
            )

        out_dict = {
            "gail_loss": self.model.loss,
            "update_batch": self.model.update_batch,
            "policy_estimate": self.model.policy_estimate,
            "expert_estimate": self.model.expert_estimate,
        }
        if self.model.use_vail:
            out_dict["kl_loss"] = self.model.kl_loss
            out_dict["z_log_sigma_sq"] = self.model.z_log_sigma_sq
            out_dict["z_mean_expert"] = self.model.z_mean_expert
            out_dict["z_mean_policy"] = self.model.z_mean_policy

        run_out = self.policy.sess.run(out_dict, feed_dict=feed_dict)
        if self.model.use_vail:
            self.update_beta(run_out["kl_loss"])
        return run_out

    def update_beta(self, kl_div: float) -> None:
        """
        Updates the Beta parameter with the latest kl_divergence value.
        The larger Beta, the stronger the importance of the kl divergence in the loss function.
        :param kl_div: The KL divergence
        """
        self.policy.sess.run(
            self.model.update_beta, feed_dict={self.model.kl_div_input: kl_div}
        )

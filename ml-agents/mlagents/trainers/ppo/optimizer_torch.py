from typing import Any, Dict
import numpy as np
import torch

from mlagents.trainers.buffer import AgentBuffer

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer


class PPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_params: Dict[str, Any]):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TFPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super(PPOOptimizer, self).__init__(policy, trainer_params)
        params = list(self.policy.actor.parameters()) + list(
            self.policy.critic.parameters()
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_params["learning_rate"]
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

    def ppo_value_loss(self, values, old_values, returns):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param returns:
        :param old_values:
        :param values:
        """

        decay_epsilon = self.trainer_params["epsilon"]

        value_losses = []
        for name, head in values.items():
            old_val_tensor = torch.DoubleTensor(old_values[name])
            clipped_value_estimate = old_val_tensor + torch.clamp(
                torch.sum(head, dim=1) - old_val_tensor, -decay_epsilon, decay_epsilon
            )
            v_opt_a = (torch.DoubleTensor(returns[name]) - torch.sum(head, dim=1)) ** 2
            v_opt_b = (torch.DoubleTensor(returns[name]) - clipped_value_estimate) ** 2
            value_loss = torch.mean(torch.max(v_opt_a, v_opt_b))
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def ppo_policy_loss(self, advantages, probs, old_probs, masks):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param masks:
        :param advantages:
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        """
        advantage = torch.from_numpy(np.expand_dims(advantages, -1))

        decay_epsilon = self.trainer_params["epsilon"]

        r_theta = torch.exp(probs - torch.DoubleTensor(old_probs))
        p_opt_a = r_theta * advantage
        p_opt_b = (
            torch.clamp(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * advantage
        )
        policy_loss = -torch.mean(torch.min(p_opt_a, p_opt_b))
        return policy_loss

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        returns = {}
        old_values = {}
        for name in self.reward_signals:
            returns[name] = batch["{}_returns".format(name)]
            old_values[name] = batch["{}_value_estimates".format(name)]

        vec_obs = np.array(batch["vector_obs"])
        vis_obs = np.array(batch["visual_obs"])
        actions, log_probs, entropy, values = self.policy.execute_model(
            vec_obs, vis_obs
        )
        value_loss = self.ppo_value_loss(values, old_values, returns)
        policy_loss = self.ppo_policy_loss(
            np.array(batch["advantages"]),
            log_probs,
            np.array(batch["action_probs"]),
            np.array(batch["masks"], dtype=np.uint32),
        )
        loss = (
            policy_loss
            + 0.5 * value_loss
            - self.trainer_params["beta"] * torch.mean(entropy)
        )
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        update_stats = {
            "Losses/Policy Loss": abs(policy_loss.detach().numpy()),
            "Losses/Value Loss": value_loss.detach().numpy(),
        }

        return update_stats

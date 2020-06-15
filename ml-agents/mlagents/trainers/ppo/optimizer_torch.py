from typing import Dict, cast
import torch

from mlagents.trainers.buffer import AgentBuffer

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import TrainerSettings, PPOSettings


class TorchPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TFPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super(TorchPPOOptimizer, self).__init__(policy, trainer_settings)
        params = list(self.policy.actor_critic.parameters())
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
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

        decay_epsilon = self.hyperparameters.epsilon

        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + torch.clamp(
                head - old_val_tensor, -decay_epsilon, decay_epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = torch.mean(torch.max(v_opt_a, v_opt_b))
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def ppo_policy_loss(self, advantages, log_probs, old_log_probs, masks):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param masks:
        :param advantages:
        :param log_probs: Current policy probabilities
        :param old_log_probs: Past policy probabilities
        """
        advantage = advantages.unsqueeze(-1)

        decay_epsilon = self.hyperparameters.epsilon

        r_theta = torch.exp(log_probs - old_log_probs)
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
            old_values[name] = torch.as_tensor(batch["{}_value_estimates".format(name)])
            returns[name] = torch.as_tensor(batch["{}_returns".format(name)])

        vec_obs = [torch.as_tensor(batch["vector_obs"])]
        act_masks = torch.as_tensor(batch["action_mask"])
        if self.policy.use_continuous_act:
            actions = torch.as_tensor(batch["actions"]).unsqueeze(-1)
        else:
            actions = torch.as_tensor(batch["actions"])

        memories = [
            torch.as_tensor(batch["memory"][i])
            for i in range(0, len(batch["memory"]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        if self.policy.use_vis_obs:
            vis_obs = []
            for idx, _ in enumerate(
                self.policy.actor_critic.network_body.visual_encoders
            ):
                vis_ob = torch.as_tensor(batch["visual_obs%d" % idx])
                vis_obs.append(vis_ob)
        else:
            vis_obs = []
        log_probs, entropy, values = self.policy.evaluate_actions(
            vec_obs,
            vis_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        value_loss = self.ppo_value_loss(values, old_values, returns)
        policy_loss = self.ppo_policy_loss(
            torch.as_tensor(batch["advantages"]),
            log_probs,
            torch.as_tensor(batch["action_probs"]),
            torch.as_tensor(batch["masks"], dtype=torch.int32),
        )
        loss = (
            policy_loss
            + 0.5 * value_loss
            - self.hyperparameters.beta * torch.mean(entropy)
        )
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        update_stats = {
            "Losses/Policy Loss": abs(policy_loss.detach().numpy()),
            "Losses/Value Loss": value_loss.detach().numpy(),
        }

        return update_stats

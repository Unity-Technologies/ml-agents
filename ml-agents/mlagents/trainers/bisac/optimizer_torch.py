import numpy as np
from typing import Dict, List, NamedTuple, cast, Tuple, Optional
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class BiSACSettings(SACSettings):
    # Add any BiSAC-specific hyperparameters here
    # For now, it inherits all from SACSettings
    pass


class TorchBiSACOptimizer(TorchSACOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        # Initialize target_q_network
        self.target_q_network = TorchSACOptimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy.network_settings,
            self._action_spec,
        )
        ModelUtils.soft_update(self.q_network, self.target_q_network, 1.0)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        value_memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]

        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
            value_memories = torch.stack(value_memories_list).unsqueeze(0)
        else:
            memories = None
            value_memories = None

        q_memories = (
            torch.zeros_like(value_memories) if value_memories is not None else None
        )

        # Copy normalizers from policy
        self.q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self._critic.network_body.copy_normalization(self.policy.actor.network_body)

        # Policy Update (BiSAC specific)
        sampled_actions, run_out, _ = self.policy.actor.get_action_and_stats(
            current_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        continuous_entropy = run_out["continuous_entropy"]
        discrete_entropy = run_out["discrete_entropy"]
        log_probs = run_out["log_probs"]

        # Q-network updates (similar to SAC)
        value_estimates, _ = self._critic.critic_pass(
            current_obs, value_memories, sequence_length=self.policy.sequence_length
        )

        cont_sampled_actions = sampled_actions.continuous_tensor
        cont_actions = actions.continuous_tensor
        q1p_out, q2p_out = self.q_network(
            current_obs,
            cont_sampled_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
            q2_grad=False,
        )
        q1_out, q2_out = self.q_network(
            current_obs,
            cont_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )

        if self._action_spec.discrete_size > 0:
            disc_actions = actions.discrete_tensor
            q1_stream = self._condense_q_streams(q1_out, disc_actions)
            q2_stream = self._condense_q_streams(q2_out, disc_actions)
        else:
            q1_stream, q2_stream = q1_out, q2_out

        with torch.no_grad():
            if value_memories is not None:
                just_first_obs = [
                    _obs[:: self.policy.sequence_length] for _obs in current_obs
                ]
                _, next_value_memories = self._critic.critic_pass(
                    just_first_obs, value_memories, sequence_length=1
                )
            else:
                next_value_memories = None
            target_values, _ = self.target_network(
                next_obs,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )
        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, target_values, dones, rewards, masks
        )
        value_loss = self.sac_value_loss(
            log_probs, value_estimates, q1p_out, q2p_out, masks
        )

        # BiSAC Policy Loss: Combine Reverse KL and Forward KL inspired components
        min_policy_qs_list = []
        for name in q1p_out.keys():
            min_policy_qs_list.append(torch.min(q1p_out[name], q2p_out[name]))
        min_policy_qs = torch.mean(torch.stack(min_policy_qs_list), dim=0)
        policy_loss_reverse_kl = ModelUtils.masked_mean(
            (self._log_ent_coef.continuous.exp() * log_probs.continuous_tensor if log_probs.continuous_tensor is not None else 0) +
            (torch.sum(self._log_ent_coef.discrete.exp() * log_probs.all_discrete_tensor, dim=1) if log_probs.all_discrete_tensor is not None else 0) -
            min_policy_qs,
            masks
        )

        # Heuristic for Forward KL: Directly maximize Q-value of sampled actions
        # This encourages the policy to produce actions that lead to high Q-values.
        # Note: This is a simplification and not a direct implementation of BiSAC's forward KL.
        sampled_actions_for_q, run_out_policy, _ = self.policy.actor.get_action_and_stats(
            current_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        continuous_entropy_policy = run_out_policy["continuous_entropy"]
        discrete_entropy_policy = run_out_policy["discrete_entropy"]
        q1_sampled, q2_sampled = self.q_network(
            current_obs,
            sampled_actions_for_q.continuous_tensor,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )
        min_q_sampled_list = []
        for name in q1_sampled.keys():
            min_q_sampled_list.append(torch.min(q1_sampled[name], q2_sampled[name]))
        min_q_sampled = torch.mean(torch.stack(min_q_sampled_list), dim=0)
        policy_loss_forward_kl_heuristic = -ModelUtils.masked_mean(min_q_sampled, masks)

        # Combine the two policy loss components
        policy_loss = policy_loss_reverse_kl + policy_loss_forward_kl_heuristic

        entropy_loss = self.sac_entropy_loss(log_probs, masks)

        total_value_loss = q1_loss + q2_loss + value_loss

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # Update target network
        ModelUtils.soft_update(self._critic, self.target_network, self.tau)
        ModelUtils.soft_update(self.q_network, self.target_q_network, self.tau)
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Learning Rate": decay_lr,
        }
        if continuous_entropy is not None:
            update_stats["Policy/Continuous Entropy"] = torch.mean(continuous_entropy).item()
        if discrete_entropy is not None:
            update_stats["Policy/Discrete Entropy"] = torch.mean(discrete_entropy).item()

        return update_stats

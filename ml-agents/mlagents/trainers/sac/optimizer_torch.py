import numpy as np
from typing import Dict, List, Mapping, cast, Tuple, Optional
from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.networks import ValueNetwork
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.timers import timed
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, SACSettings
from contextlib import ExitStack

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


class TorchSACOptimizer(TorchOptimizer):
    class PolicyValueNetwork(nn.Module):
        def __init__(
            self,
            stream_names: List[str],
            observation_shapes: List[Tuple[int, ...]],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            super().__init__()
            self.action_spec = action_spec
            if self.action_spec.is_continuous():
                self.act_size = self.action_spec.continuous_size
                num_value_outs = 1
                num_action_ins = self.act_size

            else:
                self.act_size = self.action_spec.discrete_branches
                num_value_outs = sum(self.act_size)
                num_action_ins = 0
            self.q1_network = ValueNetwork(
                stream_names,
                observation_shapes,
                network_settings,
                num_action_ins,
                num_value_outs,
            )
            self.q2_network = ValueNetwork(
                stream_names,
                observation_shapes,
                network_settings,
                num_action_ins,
                num_value_outs,
            )

        def forward(
            self,
            vec_inputs: List[torch.Tensor],
            vis_inputs: List[torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
            q1_grad: bool = True,
            q2_grad: bool = True,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            """
            Performs a forward pass on the value network, which consists of a Q1 and Q2
            network. Optionally does not evaluate gradients for either the Q1, Q2, or both.
            :param vec_inputs: List of vector observation tensors.
            :param vis_input: List of visual observation tensors.
            :param actions: For a continuous Q function (has actions), tensor of actions.
                Otherwise, None.
            :param memories: Initial memories if using memory. Otherwise, None.
            :param sequence_length: Sequence length if using memory.
            :param q1_grad: Whether or not to compute gradients for the Q1 network.
            :param q2_grad: Whether or not to compute gradients for the Q2 network.
            :return: Tuple of two dictionaries, which both map {reward_signal: Q} for Q1 and Q2,
                respectively.
            """
            # ExitStack allows us to enter the torch.no_grad() context conditionally
            with ExitStack() as stack:
                if not q1_grad:
                    stack.enter_context(torch.no_grad())
                q1_out, _ = self.q1_network(
                    vec_inputs,
                    vis_inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            with ExitStack() as stack:
                if not q2_grad:
                    stack.enter_context(torch.no_grad())
                q2_out, _ = self.q2_network(
                    vec_inputs,
                    vis_inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            return q1_out, q2_out

    def __init__(self, policy: TorchPolicy, trainer_params: TrainerSettings):
        super().__init__(policy, trainer_params)
        hyperparameters: SACSettings = cast(SACSettings, trainer_params.hyperparameters)
        self.tau = hyperparameters.tau
        self.init_entcoef = hyperparameters.init_entcoef

        self.policy = policy
        self.act_size = policy.act_size
        policy_network_settings = policy.network_settings

        self.tau = hyperparameters.tau
        self.burn_in_ratio = 0.0

        # Non-exposed SAC parameters
        self.discrete_target_entropy_scale = 0.2  # Roughly equal to e-greedy 0.05
        self.continuous_target_entropy_scale = 1.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_params.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }

        self.value_network = TorchSACOptimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_shapes,
            policy_network_settings,
            self.policy.behavior_spec.action_spec,
        )

        self.target_network = ValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_shapes,
            policy_network_settings,
        )
        ModelUtils.soft_update(
            self.policy.actor_critic.critic, self.target_network, 1.0
        )

        self._log_ent_coef = torch.nn.Parameter(
            torch.log(torch.as_tensor([self.init_entcoef] * len(self.act_size))),
            requires_grad=True,
        )
        if self.policy.use_continuous_act:
            self.target_entropy = torch.as_tensor(
                -1
                * self.continuous_target_entropy_scale
                * np.prod(self.act_size[0]).astype(np.float32)
            )
        else:
            self.target_entropy = [
                self.discrete_target_entropy_scale * np.log(i).astype(np.float32)
                for i in self.act_size
            ]

        policy_params = list(self.policy.actor_critic.network_body.parameters()) + list(
            self.policy.actor_critic.distribution.parameters()
        )
        value_params = list(self.value_network.parameters()) + list(
            self.policy.actor_critic.critic.parameters()
        )

        logger.debug("value_vars")
        for param in value_params:
            logger.debug(param.shape)
        logger.debug("policy_vars")
        for param in policy_params:
            logger.debug(param.shape)

        self.decay_learning_rate = ModelUtils.DecayedValue(
            hyperparameters.learning_rate_schedule,
            hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=hyperparameters.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            value_params, lr=hyperparameters.learning_rate
        )
        self.entropy_optimizer = torch.optim.Adam(
            [self._log_ent_coef], lr=hyperparameters.learning_rate
        )
        self._move_to_device(default_device())

    def _move_to_device(self, device: torch.device) -> None:
        self._log_ent_coef.to(device)
        self.target_network.to(device)
        self.value_network.to(device)

    def sac_q_loss(
        self,
        q1_out: Dict[str, torch.Tensor],
        q2_out: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        rewards: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(q1_out.keys()):
            q1_stream = q1_out[name].squeeze()
            q2_stream = q2_out[name].squeeze()
            with torch.no_grad():
                q_backup = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[i]
                    * target_values[name]
                )
            _q1_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q_backup, q1_stream), loss_masks
            )
            _q2_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q_backup, q2_stream), loss_masks
            )

            q1_losses.append(_q1_loss)
            q2_losses.append(_q2_loss)
        q1_loss = torch.mean(torch.stack(q1_losses))
        q2_loss = torch.mean(torch.stack(q2_losses))
        return q1_loss, q2_loss

    def sac_value_loss(
        self,
        log_probs: torch.Tensor,
        values: Dict[str, torch.Tensor],
        q1p_out: Dict[str, torch.Tensor],
        q2p_out: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        discrete: bool,
    ) -> torch.Tensor:
        min_policy_qs = {}
        with torch.no_grad():
            _ent_coef = torch.exp(self._log_ent_coef)
            for name in values.keys():
                if not discrete:
                    min_policy_qs[name] = torch.min(q1p_out[name], q2p_out[name])
                else:
                    action_probs = log_probs.exp()
                    _branched_q1p = ModelUtils.break_into_branches(
                        q1p_out[name] * action_probs, self.act_size
                    )
                    _branched_q2p = ModelUtils.break_into_branches(
                        q2p_out[name] * action_probs, self.act_size
                    )
                    _q1p_mean = torch.mean(
                        torch.stack(
                            [
                                torch.sum(_br, dim=1, keepdim=True)
                                for _br in _branched_q1p
                            ]
                        ),
                        dim=0,
                    )
                    _q2p_mean = torch.mean(
                        torch.stack(
                            [
                                torch.sum(_br, dim=1, keepdim=True)
                                for _br in _branched_q2p
                            ]
                        ),
                        dim=0,
                    )

                    min_policy_qs[name] = torch.min(_q1p_mean, _q2p_mean)

        value_losses = []
        if not discrete:
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.sum(
                        _ent_coef * log_probs, dim=1
                    )
                value_loss = 0.5 * ModelUtils.masked_mean(
                    torch.nn.functional.mse_loss(values[name], v_backup), loss_masks
                )
                value_losses.append(value_loss)
        else:
            branched_per_action_ent = ModelUtils.break_into_branches(
                log_probs * log_probs.exp(), self.act_size
            )
            # We have to do entropy bonus per action branch
            branched_ent_bonus = torch.stack(
                [
                    torch.sum(_ent_coef[i] * _lp, dim=1, keepdim=True)
                    for i, _lp in enumerate(branched_per_action_ent)
                ]
            )
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.mean(
                        branched_ent_bonus, axis=0
                    )
                value_loss = 0.5 * ModelUtils.masked_mean(
                    torch.nn.functional.mse_loss(values[name], v_backup.squeeze()),
                    loss_masks,
                )
                value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        if torch.isinf(value_loss).any() or torch.isnan(value_loss).any():
            raise UnityTrainerException("Inf found")
        return value_loss

    def sac_policy_loss(
        self,
        log_probs: torch.Tensor,
        q1p_outs: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        discrete: bool,
    ) -> torch.Tensor:
        _ent_coef = torch.exp(self._log_ent_coef)
        mean_q1 = torch.mean(torch.stack(list(q1p_outs.values())), axis=0)
        if not discrete:
            mean_q1 = mean_q1.unsqueeze(1)
            batch_policy_loss = torch.mean(_ent_coef * log_probs - mean_q1, dim=1)
            policy_loss = ModelUtils.masked_mean(batch_policy_loss, loss_masks)
        else:
            action_probs = log_probs.exp()
            branched_per_action_ent = ModelUtils.break_into_branches(
                log_probs * action_probs, self.act_size
            )
            branched_q_term = ModelUtils.break_into_branches(
                mean_q1 * action_probs, self.act_size
            )
            branched_policy_loss = torch.stack(
                [
                    torch.sum(_ent_coef[i] * _lp - _qt, dim=1, keepdim=True)
                    for i, (_lp, _qt) in enumerate(
                        zip(branched_per_action_ent, branched_q_term)
                    )
                ],
                dim=1,
            )
            batch_policy_loss = torch.squeeze(branched_policy_loss)
            policy_loss = ModelUtils.masked_mean(batch_policy_loss, loss_masks)
        return policy_loss

    def sac_entropy_loss(
        self, log_probs: torch.Tensor, loss_masks: torch.Tensor, discrete: bool
    ) -> torch.Tensor:
        if not discrete:
            with torch.no_grad():
                target_current_diff = torch.sum(log_probs + self.target_entropy, dim=1)
            entropy_loss = -1 * ModelUtils.masked_mean(
                self._log_ent_coef * target_current_diff, loss_masks
            )
        else:
            with torch.no_grad():
                branched_per_action_ent = ModelUtils.break_into_branches(
                    log_probs * log_probs.exp(), self.act_size
                )
                target_current_diff_branched = torch.stack(
                    [
                        torch.sum(_lp, axis=1, keepdim=True) + _te
                        for _lp, _te in zip(
                            branched_per_action_ent, self.target_entropy
                        )
                    ],
                    axis=1,
                )
                target_current_diff = torch.squeeze(
                    target_current_diff_branched, axis=2
                )
            entropy_loss = -1 * ModelUtils.masked_mean(
                torch.mean(self._log_ent_coef * target_current_diff, axis=1), loss_masks
            )

        return entropy_loss

    def _condense_q_streams(
        self, q_output: Dict[str, torch.Tensor], discrete_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        condensed_q_output = {}
        onehot_actions = ModelUtils.actions_to_onehot(discrete_actions, self.act_size)
        for key, item in q_output.items():
            branched_q = ModelUtils.break_into_branches(item, self.act_size)
            only_action_qs = torch.stack(
                [
                    torch.sum(_act * _q, dim=1, keepdim=True)
                    for _act, _q in zip(onehot_actions, branched_q)
                ]
            )

            condensed_q_output[key] = torch.mean(only_action_qs, dim=0)
        return condensed_q_output

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param batch: Experience mini-batch.
        :param update_target: Whether or not to update target value network
        :param reward_signal_batches: Minibatches to use for updating the reward signals,
            indexed by name. If none, don't update the reward signals.
        :return: Output from update process.
        """
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(batch[f"{name}_rewards"])

        vec_obs = [ModelUtils.list_to_tensor(batch["vector_obs"])]
        next_vec_obs = [ModelUtils.list_to_tensor(batch["next_vector_in"])]
        act_masks = ModelUtils.list_to_tensor(batch["action_mask"])
        if self.policy.use_continuous_act:
            actions = ModelUtils.list_to_tensor(batch["actions"]).unsqueeze(-1)
        else:
            actions = ModelUtils.list_to_tensor(batch["actions"], dtype=torch.long)

        memories_list = [
            ModelUtils.list_to_tensor(batch["memory"][i])
            for i in range(0, len(batch["memory"]), self.policy.sequence_length)
        ]
        # LSTM shouldn't have sequence length <1, but stop it from going out of the index if true.
        offset = 1 if self.policy.sequence_length > 1 else 0
        next_memories_list = [
            ModelUtils.list_to_tensor(
                batch["memory"][i][self.policy.m_size // 2 :]
            )  # only pass value part of memory to target network
            for i in range(offset, len(batch["memory"]), self.policy.sequence_length)
        ]

        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
            next_memories = torch.stack(next_memories_list).unsqueeze(0)
        else:
            memories = None
            next_memories = None
        # Q network memories are 0'ed out, since we don't have them during inference.
        q_memories = (
            torch.zeros_like(next_memories) if next_memories is not None else None
        )

        vis_obs: List[torch.Tensor] = []
        next_vis_obs: List[torch.Tensor] = []
        if self.policy.use_vis_obs:
            vis_obs = []
            for idx, _ in enumerate(
                self.policy.actor_critic.network_body.visual_processors
            ):
                vis_ob = ModelUtils.list_to_tensor(batch["visual_obs%d" % idx])
                vis_obs.append(vis_ob)
                next_vis_ob = ModelUtils.list_to_tensor(
                    batch["next_visual_obs%d" % idx]
                )
                next_vis_obs.append(next_vis_ob)

        # Copy normalizers from policy
        self.value_network.q1_network.network_body.copy_normalization(
            self.policy.actor_critic.network_body
        )
        self.value_network.q2_network.network_body.copy_normalization(
            self.policy.actor_critic.network_body
        )
        self.target_network.network_body.copy_normalization(
            self.policy.actor_critic.network_body
        )
        (sampled_actions, _, log_probs, _, _) = self.policy.sample_actions(
            vec_obs,
            vis_obs,
            masks=act_masks,
            memories=memories,
            seq_len=self.policy.sequence_length,
            all_log_probs=not self.policy.use_continuous_act,
        )
        value_estimates, _ = self.policy.actor_critic.critic_pass(
            vec_obs, vis_obs, memories, sequence_length=self.policy.sequence_length
        )
        if self.policy.use_continuous_act:
            squeezed_actions = actions.squeeze(-1)
            # Only need grad for q1, as that is used for policy.
            q1p_out, q2p_out = self.value_network(
                vec_obs,
                vis_obs,
                sampled_actions,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
                q2_grad=False,
            )
            q1_out, q2_out = self.value_network(
                vec_obs,
                vis_obs,
                squeezed_actions,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
            )
            q1_stream, q2_stream = q1_out, q2_out
        else:
            # For discrete, you don't need to backprop through the Q for the policy
            q1p_out, q2p_out = self.value_network(
                vec_obs,
                vis_obs,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
                q1_grad=False,
                q2_grad=False,
            )
            q1_out, q2_out = self.value_network(
                vec_obs,
                vis_obs,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
            )
            q1_stream = self._condense_q_streams(q1_out, actions)
            q2_stream = self._condense_q_streams(q2_out, actions)

        with torch.no_grad():
            target_values, _ = self.target_network(
                next_vec_obs,
                next_vis_obs,
                memories=next_memories,
                sequence_length=self.policy.sequence_length,
            )
        masks = ModelUtils.list_to_tensor(batch["masks"], dtype=torch.bool)
        use_discrete = not self.policy.use_continuous_act
        dones = ModelUtils.list_to_tensor(batch["done"])

        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, target_values, dones, rewards, masks
        )
        value_loss = self.sac_value_loss(
            log_probs, value_estimates, q1p_out, q2p_out, masks, use_discrete
        )
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks, use_discrete)
        entropy_loss = self.sac_entropy_loss(log_probs, masks, use_discrete)

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
        ModelUtils.soft_update(
            self.policy.actor_critic.critic, self.target_network, self.tau
        )
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Entropy Coeff": torch.mean(torch.exp(self._log_ent_coef)).item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def update_reward_signals(
        self, reward_signal_minibatches: Mapping[str, AgentBuffer], num_sequences: int
    ) -> Dict[str, float]:
        update_stats: Dict[str, float] = {}
        for name, update_buffer in reward_signal_minibatches.items():
            update_stats.update(self.reward_signals[name].update(update_buffer))
        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:value_network": self.value_network,
            "Optimizer:target_network": self.target_network,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
            "Optimizer:entropy_optimizer": self.entropy_optimizer,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

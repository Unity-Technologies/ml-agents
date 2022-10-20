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

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class SACSettings(OffPolicyHyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    init_entcoef: float = 1.0
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class TorchSACOptimizer(TorchOptimizer):
    class PolicyValueNetwork(nn.Module):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            super().__init__()
            num_value_outs = max(sum(action_spec.discrete_branches), 1)
            num_action_ins = int(action_spec.continuous_size)

            self.q1_network = ValueNetwork(
                stream_names,
                observation_specs,
                network_settings,
                num_action_ins,
                num_value_outs,
            )
            self.q2_network = ValueNetwork(
                stream_names,
                observation_specs,
                network_settings,
                num_action_ins,
                num_value_outs,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
            q1_grad: bool = True,
            q2_grad: bool = True,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            """
            Performs a forward pass on the value network, which consists of a Q1 and Q2
            network. Optionally does not evaluate gradients for either the Q1, Q2, or both.
            :param inputs: List of observation tensors.
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
                    inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            with ExitStack() as stack:
                if not q2_grad:
                    stack.enter_context(torch.no_grad())
                q2_out, _ = self.q2_network(
                    inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            return q1_out, q2_out

    class TargetEntropy(NamedTuple):

        discrete: List[float] = []  # One per branch
        continuous: float = 0.0

    class LogEntCoef(nn.Module):
        def __init__(self, discrete, continuous):
            super().__init__()
            self.discrete = discrete
            self.continuous = continuous

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if isinstance(policy.actor, SharedActorCritic):
            raise UnityTrainerException("SAC does not support SharedActorCritic")
        self._critic = ValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            policy.network_settings,
        )
        hyperparameters: SACSettings = cast(
            SACSettings, trainer_settings.hyperparameters
        )

        self.tau = hyperparameters.tau
        self.init_entcoef = hyperparameters.init_entcoef

        self.policy = policy
        policy_network_settings = policy.network_settings

        self.tau = hyperparameters.tau
        self.burn_in_ratio = 0.0

        # Non-exposed SAC parameters
        self.discrete_target_entropy_scale = 0.2  # Roughly equal to e-greedy 0.05
        self.continuous_target_entropy_scale = 1.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_settings.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }
        self._action_spec = self.policy.behavior_spec.action_spec

        self.q_network = TorchSACOptimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
        )

        self.target_network = ValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
        )
        ModelUtils.soft_update(self._critic, self.target_network, 1.0)

        # We create one entropy coefficient per action, whether discrete or continuous.
        _disc_log_ent_coef = torch.nn.Parameter(
            torch.log(
                torch.as_tensor(
                    [self.init_entcoef] * len(self._action_spec.discrete_branches)
                )
            ),
            requires_grad=True,
        )
        _cont_log_ent_coef = torch.nn.Parameter(
            torch.log(torch.as_tensor([self.init_entcoef])), requires_grad=True
        )
        self._log_ent_coef = TorchSACOptimizer.LogEntCoef(
            discrete=_disc_log_ent_coef, continuous=_cont_log_ent_coef
        )
        _cont_target = (
            -1
            * self.continuous_target_entropy_scale
            * np.prod(self._action_spec.continuous_size).astype(np.float32)
        )
        _disc_target = [
            self.discrete_target_entropy_scale * np.log(i).astype(np.float32)
            for i in self._action_spec.discrete_branches
        ]
        self.target_entropy = TorchSACOptimizer.TargetEntropy(
            continuous=_cont_target, discrete=_disc_target
        )
        policy_params = list(self.policy.actor.parameters())
        value_params = list(self.q_network.parameters()) + list(
            self._critic.parameters()
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
            self._log_ent_coef.parameters(), lr=hyperparameters.learning_rate
        )
        self._move_to_device(default_device())

    @property
    def critic(self):
        return self._critic

    def _move_to_device(self, device: torch.device) -> None:
        self._log_ent_coef.to(device)
        self.target_network.to(device)
        self._critic.to(device)
        self.q_network.to(device)

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
        log_probs: ActionLogProbs,
        values: Dict[str, torch.Tensor],
        q1p_out: Dict[str, torch.Tensor],
        q2p_out: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        min_policy_qs = {}
        with torch.no_grad():
            _cont_ent_coef = self._log_ent_coef.continuous.exp()
            _disc_ent_coef = self._log_ent_coef.discrete.exp()
            for name in values.keys():
                if self._action_spec.discrete_size <= 0:
                    min_policy_qs[name] = torch.min(q1p_out[name], q2p_out[name])
                else:
                    disc_action_probs = log_probs.all_discrete_tensor.exp()
                    _branched_q1p = ModelUtils.break_into_branches(
                        q1p_out[name] * disc_action_probs,
                        self._action_spec.discrete_branches,
                    )
                    _branched_q2p = ModelUtils.break_into_branches(
                        q2p_out[name] * disc_action_probs,
                        self._action_spec.discrete_branches,
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
        if self._action_spec.discrete_size <= 0:
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.sum(
                        _cont_ent_coef * log_probs.continuous_tensor, dim=1
                    )
                value_loss = 0.5 * ModelUtils.masked_mean(
                    torch.nn.functional.mse_loss(values[name], v_backup), loss_masks
                )
                value_losses.append(value_loss)
        else:
            disc_log_probs = log_probs.all_discrete_tensor
            branched_per_action_ent = ModelUtils.break_into_branches(
                disc_log_probs * disc_log_probs.exp(),
                self._action_spec.discrete_branches,
            )
            # We have to do entropy bonus per action branch
            branched_ent_bonus = torch.stack(
                [
                    torch.sum(_disc_ent_coef[i] * _lp, dim=1, keepdim=True)
                    for i, _lp in enumerate(branched_per_action_ent)
                ]
            )
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.mean(
                        branched_ent_bonus, axis=0
                    )
                    # Add continuous entropy bonus to minimum Q
                    if self._action_spec.continuous_size > 0:
                        v_backup += torch.sum(
                            _cont_ent_coef * log_probs.continuous_tensor,
                            dim=1,
                            keepdim=True,
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
        log_probs: ActionLogProbs,
        q1p_outs: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        _cont_ent_coef, _disc_ent_coef = (
            self._log_ent_coef.continuous,
            self._log_ent_coef.discrete,
        )
        _cont_ent_coef = _cont_ent_coef.exp()
        _disc_ent_coef = _disc_ent_coef.exp()

        mean_q1 = torch.mean(torch.stack(list(q1p_outs.values())), axis=0)
        batch_policy_loss = 0
        if self._action_spec.discrete_size > 0:
            disc_log_probs = log_probs.all_discrete_tensor
            disc_action_probs = disc_log_probs.exp()
            branched_per_action_ent = ModelUtils.break_into_branches(
                disc_log_probs * disc_action_probs, self._action_spec.discrete_branches
            )
            branched_q_term = ModelUtils.break_into_branches(
                mean_q1 * disc_action_probs, self._action_spec.discrete_branches
            )
            branched_policy_loss = torch.stack(
                [
                    torch.sum(_disc_ent_coef[i] * _lp - _qt, dim=1, keepdim=False)
                    for i, (_lp, _qt) in enumerate(
                        zip(branched_per_action_ent, branched_q_term)
                    )
                ],
                dim=1,
            )
            batch_policy_loss += torch.sum(branched_policy_loss, dim=1)
            all_mean_q1 = torch.sum(disc_action_probs * mean_q1, dim=1)
        else:
            all_mean_q1 = mean_q1
        if self._action_spec.continuous_size > 0:
            cont_log_probs = log_probs.continuous_tensor
            batch_policy_loss += (
                _cont_ent_coef * torch.sum(cont_log_probs, dim=1) - all_mean_q1
            )
        policy_loss = ModelUtils.masked_mean(batch_policy_loss, loss_masks)

        return policy_loss

    def sac_entropy_loss(
        self, log_probs: ActionLogProbs, loss_masks: torch.Tensor
    ) -> torch.Tensor:
        _cont_ent_coef, _disc_ent_coef = (
            self._log_ent_coef.continuous,
            self._log_ent_coef.discrete,
        )
        entropy_loss = 0
        if self._action_spec.discrete_size > 0:
            with torch.no_grad():
                # Break continuous into separate branch
                disc_log_probs = log_probs.all_discrete_tensor
                branched_per_action_ent = ModelUtils.break_into_branches(
                    disc_log_probs * disc_log_probs.exp(),
                    self._action_spec.discrete_branches,
                )
                target_current_diff_branched = torch.stack(
                    [
                        torch.sum(_lp, axis=1, keepdim=True) + _te
                        for _lp, _te in zip(
                            branched_per_action_ent, self.target_entropy.discrete
                        )
                    ],
                    axis=1,
                )
                target_current_diff = torch.squeeze(
                    target_current_diff_branched, axis=2
                )
            entropy_loss += -1 * ModelUtils.masked_mean(
                torch.mean(_disc_ent_coef * target_current_diff, axis=1), loss_masks
            )
        if self._action_spec.continuous_size > 0:
            with torch.no_grad():
                cont_log_probs = log_probs.continuous_tensor
                target_current_diff = (
                    torch.sum(cont_log_probs, dim=1) + self.target_entropy.continuous
                )
            # We update all the _cont_ent_coef as one block
            entropy_loss += -1 * ModelUtils.masked_mean(
                _cont_ent_coef * target_current_diff, loss_masks
            )

        return entropy_loss

    def _condense_q_streams(
        self, q_output: Dict[str, torch.Tensor], discrete_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        condensed_q_output = {}
        onehot_actions = ModelUtils.actions_to_onehot(
            discrete_actions, self._action_spec.discrete_branches
        )
        for key, item in q_output.items():
            branched_q = ModelUtils.break_into_branches(
                item, self._action_spec.discrete_branches
            )
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
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        # LSTM shouldn't have sequence length <1, but stop it from going out of the index if true.
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

        # Q and V network memories are 0'ed out, since we don't have them during inference.
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
        sampled_actions, run_out, _, = self.policy.actor.get_action_and_stats(
            current_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"]
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
            # Since we didn't record the next value memories, evaluate one step in the critic to
            # get them.
            if value_memories is not None:
                # Get the first observation in each sequence
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
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks)
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
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Discrete Entropy Coeff": torch.mean(
                torch.exp(self._log_ent_coef.discrete)
            ).item(),
            "Policy/Continuous Entropy Coeff": torch.mean(
                torch.exp(self._log_ent_coef.continuous)
            ).item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:q_network": self.q_network,
            "Optimizer:value_network": self._critic,
            "Optimizer:target_network": self.target_network,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
            "Optimizer:entropy_optimizer": self.entropy_optimizer,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

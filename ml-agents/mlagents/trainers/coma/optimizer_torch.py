from typing import Dict, cast, List, Tuple, Optional
import numpy as np
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents_envs.base_env import ObservationSpec, ActionSpec
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    ExtrinsicSettings,
    RewardSignalSettings,
    RewardSignalType,
    TrainerSettings,
    PPOSettings,
)
from mlagents.trainers.torch.networks import Critic, MultiInputNetworkBody
from mlagents.trainers.torch.decoders import ValueHeads
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil, GroupObsUtil
from mlagents.trainers.settings import NetworkSettings

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class TorchCOMAOptimizer(TorchOptimizer):
    class COMAValueNetwork(torch.nn.Module, Critic):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            torch.nn.Module.__init__(self)
            self.network_body = MultiInputNetworkBody(
                observation_specs, network_settings, action_spec
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units

            self.value_heads = ValueHeads(stream_names, encoding_size, 1)

        @property
        def memory_size(self) -> int:
            return self.network_body.memory_size

        def update_normalization(self, buffer: AgentBuffer) -> None:
            self.network_body.update_normalization(buffer)

        def baseline(
            self,
            self_obs: List[List[torch.Tensor]],
            obs: List[List[torch.Tensor]],
            actions: List[AgentAction],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            encoding, memories = self.network_body(
                obs_only=self_obs,
                obs=obs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )
            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

        def critic_pass(
            self,
            obs: List[List[torch.Tensor]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )
            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

        def forward(
            self,
            encoding: torch.Tensor,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            output = self.value_heads(encoding)
            return output, memories

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self._critic = TorchCOMAOptimizer.COMAValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            network_settings=trainer_settings.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )

        params = list(self.policy.actor.parameters()) + list(self.critic.parameters())
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

    def create_reward_signals(
        self, reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]
    ) -> None:
        """
        Create reward signals. Override default to provide warnings for Curiosity and
        GAIL, and make sure Extrinsic adds team rewards.
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal, settings in reward_signal_configs.items():
            if reward_signal != RewardSignalType.EXTRINSIC:
                logger.warning(
                    f"Reward Signal {reward_signal.value} is not supported with the COMA2 trainer; \
                    results may be unexpected."
                )
            elif isinstance(settings, ExtrinsicSettings):
                settings.add_groupmate_rewards = True
        super().create_reward_signals(reward_signal_configs)

    @property
    def critic(self):
        return self._critic

    def coma_value_loss(
        self,
        values: Dict[str, torch.Tensor],
        old_values: Dict[str, torch.Tensor],
        returns: Dict[str, torch.Tensor],
        epsilon: float,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates value loss for PPO.
        :param values: Value output of the current network.
        :param old_values: Value stored with experiences in buffer.
        :param returns: Computed returns.
        :param epsilon: Clipping value for value estimate.
        :param loss_mask: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + torch.clamp(
                head - old_val_tensor, -1 * epsilon, epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = ModelUtils.masked_mean(torch.max(v_opt_a, v_opt_b), loss_masks)
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def ppo_policy_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate PPO policy loss.
        :param advantages: Computed advantages.
        :param log_probs: Current policy probabilities
        :param old_log_probs: Past policy probabilities
        :param loss_masks: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        advantage = advantages.unsqueeze(-1)

        decay_epsilon = self.hyperparameters.epsilon
        r_theta = torch.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = (
            torch.clamp(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * advantage
        )
        policy_loss = -1 * ModelUtils.masked_mean(
            torch.min(p_opt_a, p_opt_b), loss_masks
        )
        return policy_loss

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        old_baseline_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )
            old_baseline_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.baseline_estimates_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        group_obs = GroupObsUtil.from_buffer(batch, n_obs)
        group_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs]
            for _groupmate_obs in group_obs
        ]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)
        group_actions = AgentAction.group_from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        log_probs, entropy = self.policy.evaluate_actions(
            current_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        all_obs = [current_obs] + group_obs
        values, _ = self.critic.critic_pass(
            all_obs, memories=memories, sequence_length=self.policy.sequence_length
        )
        baselines, _ = self.critic.baseline(
            [current_obs],
            group_obs,
            group_actions,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        baseline_loss = self.coma_value_loss(
            baselines, old_baseline_values, returns, decay_eps, loss_masks
        )
        value_loss = self.coma_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = self.ppo_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
        )
        loss = (
            policy_loss
            + 0.5 * (value_loss + 0.5 * baseline_loss)
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Baseline Loss": baseline_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))

        return update_stats

    def get_modules(self):
        modules = {"Optimizer": self.optimizer}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    def get_trajectory_and_baseline_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        next_group_obs: List[List[np.ndarray]],
        done: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:

        n_obs = len(self.policy.behavior_spec.observation_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        team_obs = GroupObsUtil.from_buffer(batch, n_obs)

        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        team_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _teammate_obs]
            for _teammate_obs in team_obs
        ]

        team_actions = AgentAction.group_from_buffer(batch)

        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        next_group_obs = [
            ModelUtils.list_to_tensor_list(_list_obs) for _list_obs in next_group_obs
        ]
        # Expand dimensions of next critic obs
        next_group_obs = [
            [_obs.unsqueeze(0) for _obs in _list_obs] for _list_obs in next_group_obs
        ]

        memory = torch.zeros([1, 1, self.policy.m_size])
        all_obs = [current_obs] + team_obs if team_obs is not None else [current_obs]
        value_estimates, mem = self.critic.critic_pass(
            all_obs, memory, sequence_length=batch.num_experiences
        )

        baseline_estimates, mem = self.critic.baseline(
            [current_obs],
            team_obs,
            team_actions,
            memory,
            sequence_length=batch.num_experiences,
        )
        all_next_obs = (
            [next_obs] + next_group_obs if next_group_obs is not None else [next_obs]
        )

        next_value_estimates, mem = self.critic.critic_pass(
            all_next_obs, mem, sequence_length=batch.num_experiences
        )

        for name, estimate in baseline_estimates.items():
            baseline_estimates[name] = ModelUtils.to_numpy(estimate)

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)

        # the base line and V shpuld  not be on the same done flag
        for name, estimate in next_value_estimates.items():
            next_value_estimates[name] = ModelUtils.to_numpy(estimate)

        if done:
            for k in next_value_estimates:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimates[k][-1] = 0.0

        return (value_estimates, baseline_estimates, next_value_estimates)

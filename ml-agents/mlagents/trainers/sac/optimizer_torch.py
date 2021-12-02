import numpy as np
from typing import Dict, List, Mapping, NamedTuple, cast, Tuple, Optional
from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.networks import ValueNetwork
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.distributions import GaussianDistInstance
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, SACSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)

from mlagents.trainers.torch.action_flattener import ActionFlattener
from mlagents_envs.base_env import ObservationType
from mlagents.trainers.torch.networks import NetworkBody
from mlagents_envs.base_env import BehaviorSpec

from mlagents.trainers.torch.layers import linear_layer, Initialization


class DiverseNetworkVariational(torch.nn.Module):
    alpha = 0.0005
    MIN_STDDEV = 0.1
    EPSILON = 1e-7

    def __init__(
            self, 
            specs: BehaviorSpec, 
            use_actions: bool = True, 
            continuous: bool = False, 
            learn_stddev: bool = True, 
            noise: bool = True, 
            stddev_param: bool = False,
            centered_reward: bool = False,
            dropout: float = 0.2,
            name: str = "MEDE"
        ) -> None:
        super().__init__()
        self._use_actions = use_actions
        self._continuous = continuous
        self._learn_stddev = learn_stddev
        self._centered_reward = centered_reward
        self._name = name
        self.stats = {}
        sigma_start = 0.5
        stddev_start = 1.0

        encoder_settings = NetworkSettings(normalize=True, num_layers=1)
        if encoder_settings.memory is not None:
            encoder_settings.memory = None
            logger.warning(
                "memory was specified in network_settings but is not supported. It is being ignored."
            )

        new_spec = [
            spec
            for spec in specs.observation_specs
            if spec.observation_type != ObservationType.GOAL_SIGNAL
        ]
        diverse_spec = [
            spec
            for spec in specs.observation_specs
            if spec.observation_type == ObservationType.GOAL_SIGNAL
        ][0]
        self._all_obs_specs = specs.observation_specs
        self.diverse_size = diverse_spec.shape[0]

        self._dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        
        self.disc_sizes = specs.action_spec.discrete_branches
        self.cont_size = specs.action_spec.continuous_size

        if self._use_actions and self.cont_size > 0:
            self._encoder = NetworkBody(
                new_spec, encoder_settings, self.cont_size
            )
        else:
            self._encoder = NetworkBody(new_spec, encoder_settings)

        self._z_sigma = torch.nn.Parameter(
            sigma_start * torch.ones((encoder_settings.hidden_units), dtype=torch.float),
            requires_grad=True,
        ) if noise else None
        self._stddev = torch.nn.Parameter(
            torch.tensor(stddev_start, dtype=torch.float), requires_grad=True
        ) if self._learn_stddev and self._continuous and stddev_param else None

        self._out_size = self.diverse_size * 2 if self._learn_stddev and \
                                                  self._continuous and \
                                                  not stddev_param \
                                               else self.diverse_size
        disc_actions = sum(self.disc_sizes) if self._use_actions and len(self.disc_sizes) > 0 else 1
        self._last_layer = torch.nn.Linear(encoder_settings.hidden_units, self._out_size * disc_actions)

        self._diverse_index = -1
        self._max_index = len(specs.observation_specs)
        for i, spec in enumerate(specs.observation_specs):
            if spec.observation_type == ObservationType.GOAL_SIGNAL:
                self._diverse_index = i

    def predict(
        self, obs_input, action, detach_action=False, var_noise=True
    ) -> torch.Tensor:
        # Convert to tensors
        tensor_obs = [
            obs
            for obs, spec in zip(obs_input, self._all_obs_specs)
            if spec.observation_type != ObservationType.GOAL_SIGNAL
        ]
        if self._use_actions and self.cont_size > 0:
            if detach_action:
                action = action.detach()
            hidden, _ = self._encoder.forward(tensor_obs, action)
        else:
            hidden, _ = self._encoder.forward(tensor_obs)

        if self._dropout is not None:
            hidden = self._dropout(hidden)

        #z_mu = hidden
        #if self._z_sigma is not None and var_noise:
        #    hidden = z_mu + torch.randn_like(z_mu) * self._z_sigma

        prediction = self._last_layer(hidden)

        return prediction, hidden

    def copy_normalization(self, thing):
        j = 0
        for i, spec in enumerate(self._all_obs_specs):
            if spec.observation_type != ObservationType.GOAL_SIGNAL:
                self._encoder.processors[j].copy_normalization(thing.processors[i])
                j+=1

    def _get_log_probs(self, pred, truth, masks=None):
        if self._continuous:
            if self._learn_stddev and self._stddev is None:
                mean = torch.tanh(pred[:, :int(pred.shape[1] / 2)])
                std = torch.clamp(torch.exp(pred[:, int(pred.shape[1] / 2):]), self.MIN_STDDEV, 5)
            elif self._learn_stddev:
                mean = torch.tanh(pred)
                std = torch.ones_like(mean) * torch.clamp(torch.exp(self._stddev), self.MIN_STDDEV, 5)
            else:
                mean = torch.tanh(pred)
                std = torch.full_like(pred, 0.1)

            dist = GaussianDistInstance(mean, std)
            max_logprob = torch.sum(
                GaussianDistInstance(
                    torch.zeros(self.diverse_size), torch.ones(self.diverse_size) * self.MIN_STDDEV
                ).log_prob(torch.zeros(self.diverse_size))
            )
            log_probs = torch.sum(dist.log_prob(truth), dim=1)

            if masks is not None:
                self.stats.update({
                    self._name + "/Variance of Mean": torch.mean(torch.var(mean, dim=0)).item(),
                    self._name + "/Mean of StdDev": torch.mean(std).item(),
                    self._name + "/Mean of Variance": torch.mean(std ** 2).item(),
                    self._name + "/LogProb": ModelUtils.masked_mean(log_probs, masks).item()
                })

            return log_probs - max_logprob

        else:
            probs = torch.softmax(pred, dim=1)
            log_probs = torch.log(
                torch.sum(
                    probs * truth, dim=1
                ) + self.EPSILON
            )

            if masks is not None:
                self.stats.update({
                    self._name + "/LogProb": ModelUtils.masked_mean(log_probs, masks).item()
                })

            return log_probs

    def rewards(
        self, obs_input, cont_act, disc_act, detach_action=False, var_noise=True, masks=None
    ) -> torch.Tensor:

        truth = obs_input[self._diverse_index]
        prediction, _ = self.predict(obs_input, cont_act, detach_action, var_noise)

        if self._use_actions and len(self.disc_sizes) > 0:
            
            disc_probs = disc_act.exp()
            if detach_action:
                disc_act = disc_act.detach()
                disc_probs = disc_probs.detach()

            action_rewards = []
            action_accuracy = []
            for i in range(0, sum(self.disc_sizes)*self._out_size, self._out_size):  # Get the reward for each discrete action
                action_rewards.append(
                    self._get_log_probs(prediction[:, i:i+self._out_size], truth, masks).unsqueeze(1)
                )
                if not self._continuous:
                    action_accuracy.append(
                        torch.argmax(prediction[:, i:i+self._out_size], dim=1, keepdim=True) == \
                        torch.argmax(truth, dim=1, keepdim=True)
                    )

            all_rewards = torch.cat(action_rewards, dim=1)
            branched_rewards = ModelUtils.break_into_branches(all_rewards + disc_act, self.disc_sizes)
            rewards = torch.sum(torch.stack([torch.sum(branch, dim=1) for branch in branched_rewards]), dim=0)

            accuracy = torch.zeros_like(rewards)
            if not self._continuous:
                all_accuracy = torch.cat(action_accuracy, dim=1)
                branched_accuracy = ModelUtils.break_into_branches(all_accuracy * disc_act.exp(), self.disc_sizes)
                accuracy = torch.mean(torch.stack([torch.sum(branch, dim=1) for branch in branched_accuracy]), dim=0)

        else:
            rewards = self._get_log_probs(prediction, truth, masks)
            accuracy = torch.zeros_like(rewards)
            if not self._continuous:
                accuracy = torch.argmax(prediction, dim=1) == torch.argmax(truth, dim=1)

        # Log accuracy stats
        if masks is not None:
            self.stats.update({
                self._name + "/Accuracy": ModelUtils.masked_mean(accuracy, masks).item(),
            })
            truth = torch.argmax(obs_input[self._diverse_index], dim=1)
            counts = torch.bincount(truth, minlength=self.diverse_size)
            #for i in range(self.diverse_size):
            #    a = torch.sum(torch.where(truth == i, accuracy, torch.zeros_like(accuracy))) / counts[i]
            #    self.stats.update({
            #        "Settings/MEDE Accuracy {}".format(i): a.item()
            #    })

        if self._centered_reward:
            rewards -= np.log(1 / self.diverse_size)

        return rewards

    def loss(
        self, obs_input, cont_act, disc_act, masks, detach_action=True, var_noise=True
    ) -> torch.Tensor:

        rewards = self.rewards(obs_input, cont_act, disc_act, detach_action, var_noise, masks)
        base_loss = -ModelUtils.masked_mean(rewards, masks)
        total_loss = base_loss

        self.stats.update({
            f"Losses/{self._name} Loss": total_loss.item(),
            self._name + "/Reward": -base_loss.item(),
        })

        truth = torch.argmax(obs_input[self._diverse_index], dim=1)
        counts = torch.bincount(truth, minlength=self.diverse_size)
        for i in range(self.diverse_size):
            r = torch.sum(torch.where(truth == i, rewards, torch.zeros_like(rewards))) / counts[i]
            self.stats.update({
                f"Settings/{self._name} Reward".format(i): r.item(),
            })
        
        return total_loss, self.stats


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

    class DivCoef(nn.Module):
        def __init__(self, coef):
            super().__init__()
            self.coef = coef

    def __init__(self, policy: TorchPolicy, trainer_params: TrainerSettings):
        super().__init__(policy, trainer_params)
        reward_signal_configs = trainer_params.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if policy.shared_critic:
            raise UnityTrainerException("SAC does not support SharedActorCritic")
        self._critic = ValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            policy.network_settings,
        )

        hyperparameters: SACSettings = cast(SACSettings, trainer_params.hyperparameters)
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
        self.gammas = [_val.gamma for _val in trainer_params.reward_signals.values()]
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

        # Diversity
        if hyperparameters.mede:
            self._use_mede = True
            self._use_diayn = False
        elif hyperparameters.diayn:
            self._use_mede = False
            self._use_diayn = True
        else:
            self._use_mede = False
            self._use_diayn = False

        self._mede_network = DiverseNetworkVariational(
            self.policy.behavior_spec, 
            hyperparameters.mede_use_actions,
            hyperparameters.mede_continuous,
            hyperparameters.mede_learn_stddev,
            hyperparameters.mede_noise,
            hyperparameters.mede_stddev_param,
            hyperparameters.mede_centered_reward,
            hyperparameters.mede_dropout
        ) if self._use_mede or (self._use_diayn and hyperparameters.mede_both_discriminators) else None
        self._diayn_network = DiverseNetworkVariational(
            self.policy.behavior_spec, 
            False,
            hyperparameters.mede_continuous,
            hyperparameters.mede_learn_stddev,
            hyperparameters.mede_noise,
            hyperparameters.mede_stddev_param,
            hyperparameters.mede_centered_reward,
            hyperparameters.mede_dropout,
            "DIAYN"
        ) if self._use_diayn or (self._use_mede and hyperparameters.mede_both_discriminators) else None

        self._mede_policy_loss = hyperparameters.mede_for_policy_loss
        self._target_diversity = hyperparameters.mede_target_divcoef
        self._adaptive_divcoef = hyperparameters.mede_adaptive_divcoef
        self._scheduled_divcoef = hyperparameters.mede_scheduled_divcoef
        self._delta_divcoef = hyperparameters.mede_divcoef_lr
        self._diversity_coef = TorchSACOptimizer.DivCoef(torch.nn.Parameter(
            torch.as_tensor([hyperparameters.mede_init_divcoef]), requires_grad=self._adaptive_divcoef
        ))

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
        self.mede_optimizer = torch.optim.Adam(
            list(self._mede_network.parameters()), 
            lr=hyperparameters.learning_rate, 
        ) if self._use_mede or (self._use_diayn and hyperparameters.mede_both_discriminators) else None
        self.diayn_optimizer = torch.optim.Adam(
            list(self._diayn_network.parameters()), 
            lr=hyperparameters.learning_rate, 
        ) if self._use_diayn or (self._use_mede and hyperparameters.mede_both_discriminators) else None
        self.divcoef_optimizer = torch.optim.Adam(
            self._diversity_coef.parameters(),
            lr=hyperparameters.mede_divcoef_lr, 
        ) if self._use_mede or self._use_diayn else None
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
        diayn_rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(q1_out.keys()):
            q1_stream = q1_out[name].squeeze()
            q2_stream = q2_out[name].squeeze()
            with torch.no_grad():
                q_backup = rewards[name] + \
                    self._diversity_coef.coef * diayn_rewards + (
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
        mede_rewards: torch.Tensor
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
                    v_backup += self._diversity_coef.coef * mede_rewards
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
                    v_backup += self._diversity_coef.coef * mede_rewards.unsqueeze(-1)
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
        mede_rewards: torch.Tensor
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
        if self._mede_policy_loss:
            batch_policy_loss += -self._diversity_coef.coef * mede_rewards
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
                target_current_diff = torch.sum(cont_log_probs,dim=1) + self.target_entropy.continuous
            # We update all the _cont_ent_coef as one block
            entropy_loss += -1 * ModelUtils.masked_mean(
                _cont_ent_coef * target_current_diff, loss_masks
            )

        return entropy_loss

    def sac_divcoef_loss(
        self, mede_rewards: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            mede_loss = -ModelUtils.masked_mean(mede_rewards, masks)
            target_diff = mede_loss - self._target_diversity

        divcoef_loss = -1 * (self._diversity_coef.coef * target_diff)
        return divcoef_loss

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
        sampled_actions, log_probs, _, _, = self.policy.actor.get_action_and_stats(
            current_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
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

        mede_value_rewards, mede_policy_rewards, diayn_rewards = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        if self._use_mede:
            self._mede_network.copy_normalization(self.policy.actor.network_body)
            disc_act = log_probs.all_discrete_tensor if self._action_spec.discrete_size > 0 else None
            mede_policy_rewards = self._mede_network.rewards(
                current_obs, sampled_actions.continuous_tensor, disc_act, var_noise=False
            )
            with torch.no_grad():
                mede_value_rewards = self._mede_network.rewards(
                    current_obs, sampled_actions.continuous_tensor, disc_act, var_noise=False
                )
        elif self._use_diayn:
            self._diayn_network.copy_normalization(self.policy.actor.network_body)
            disc_act = log_probs.all_discrete_tensor if self._action_spec.discrete_size > 0 else None
            with torch.no_grad():
                diayn_rewards = self._diayn_network.rewards(
                    current_obs, sampled_actions.continuous_tensor, disc_act, var_noise=False
                )

        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, target_values, dones, rewards, masks, diayn_rewards
        )
        value_loss = self.sac_value_loss(
            log_probs, value_estimates, q1p_out, q2p_out, masks, mede_value_rewards
        )
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks, mede_policy_rewards)
        entropy_loss = self.sac_entropy_loss(log_probs, masks)

        total_value_loss = q1_loss + q2_loss
        if self.policy.shared_critic:
            policy_loss += value_loss
        else:
            total_value_loss += value_loss

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        #ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)
        #self.entropy_optimizer.zero_grad()
        #entropy_loss.backward()
        #self.entropy_optimizer.step()

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
            "Policy/Entropy Loss": entropy_loss.item(),
        }

        if self._use_mede or self._use_diayn:
            disc_act = log_probs.all_discrete_tensor if self._action_spec.discrete_size > 0 else None

            if self._mede_network is not None:
                #mede_loss, mede_stats = self._mede_network.loss(
                #    current_obs, sampled_actions.continuous_tensor, disc_act, masks
                #)
                mede_loss, mede_stats = self._mede_network.loss(
                    current_obs, cont_actions, disc_act, masks
                )


                ModelUtils.update_learning_rate(self.mede_optimizer, decay_lr)
                self.mede_optimizer.zero_grad()
                mede_loss.backward()
                self.mede_optimizer.step()

                update_stats.update(mede_stats)

            if self._diayn_network is not None:
                diayn_loss, diayn_stats = self._diayn_network.loss(
                    current_obs, sampled_actions.continuous_tensor, disc_act, masks
                )

                ModelUtils.update_learning_rate(self.mede_optimizer, decay_lr)
                self.diayn_optimizer.zero_grad()
                diayn_loss.backward()
                self.diayn_optimizer.step()
            
                update_stats.update(diayn_stats)
            
            if self._adaptive_divcoef:
                divcoef_loss = self.sac_divcoef_loss(mede_value_rewards, masks)
                self.divcoef_optimizer.zero_grad()
                divcoef_loss.backward()
                self.divcoef_optimizer.step()
                update_stats.update({
                    "Policy/Diversity Coefficient": self._diversity_coef.coef.item(),
                    "Policy/Diversity Coefficient Loss": divcoef_loss.item()
                })

            elif self._scheduled_divcoef:
                if self._diversity_coef.coef < self._target_diversity:
                    self._diversity_coef.coef += self._delta_divcoef
                update_stats.update({
                    "Policy/Diversity Coefficient": self._diversity_coef.coef.item(),
                })

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
            "Optimizer:q_network": self.q_network,
            "Optimizer:value_network": self._critic,
            "Optimizer:target_network": self.target_network,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
            "Optimizer:entropy_optimizer": self.entropy_optimizer,
        }
        if self._use_mede:
            modules.update({
                "Optimizer:mede_optimizer": self.mede_optimizer,
                "Optimizer:mede_network": self._mede_network,
            })
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

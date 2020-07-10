import numpy as np
from typing import Dict, List, Mapping, cast, Tuple
import torch
from torch import nn

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.brain import CameraResolution
from mlagents.trainers.models_torch import Critic, ContinuousQNetwork, ActionType
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.timers import timed
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, SACSettings

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)

POLICY_SCOPE = ""
TARGET_SCOPE = "target_network"


class TorchSACOptimizer(TorchOptimizer):
    class PolicyValueNetwork(nn.Module):
        def __init__(
            self,
            stream_names: List[str],
            vector_sizes: List[int],
            visual_sizes: List[CameraResolution],
            network_settings: NetworkSettings,
            act_type: ActionType,
            act_size: List[int],
        ):
            super().__init__()
            if act_type == ActionType.CONTINUOUS:
                self.q1_network = ContinuousQNetwork(
                    stream_names,
                    vector_sizes,
                    visual_sizes,
                    network_settings,
                    act_type,
                    act_size,
                )
                self.q2_network = ContinuousQNetwork(
                    stream_names,
                    vector_sizes,
                    visual_sizes,
                    network_settings,
                    act_type,
                    act_size,
                )
            else:
                raise UnityTrainerException("Not supported yet")

        def forward(
            self,
            vec_inputs: List[torch.Tensor],
            vis_inputs: List[torch.Tensor],
            actions: torch.Tensor = None,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            if actions is not None:
                assert isinstance(self.q1_network, ContinuousQNetwork)
                q1_out, _ = self.q1_network(vec_inputs, vis_inputs, actions=actions)
                q2_out, _ = self.q2_network(vec_inputs, vis_inputs, actions=actions)
            return q1_out, q2_out

    def __init__(self, policy: TorchPolicy, trainer_params: TrainerSettings):
        super().__init__(policy, trainer_params)
        hyperparameters: SACSettings = cast(SACSettings, trainer_params.hyperparameters)
        lr = hyperparameters.learning_rate
        # lr_schedule = hyperparameters.learning_rate_schedule
        # max_step = trainer_params.max_steps
        self.tau = hyperparameters.tau
        self.init_entcoef = hyperparameters.init_entcoef

        self.policy = policy
        self.act_size = policy.act_size
        policy_network_settings = policy.network_settings
        # h_size = policy_network_settings.hidden_units
        # num_layers = policy_network_settings.num_layers
        # vis_encode_type = policy_network_settings.vis_encode_type

        self.tau = hyperparameters.tau
        self.burn_in_ratio = 0.0

        # Non-exposed SAC parameters
        self.discrete_target_entropy_scale = 0.2  # Roughly equal to e-greedy 0.05
        self.continuous_target_entropy_scale = 1.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_params.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(self.reward_signals[name].use_terminal_states)
            for name in self.stream_names
        }
        # self.disable_use_dones = {
        #     name: self.use_dones_in_backup[name].assign(0.0)
        #     for name in stream_names
        # }

        brain = policy.brain
        self.value_network = TorchSACOptimizer.PolicyValueNetwork(
            self.stream_names,
            [brain.vector_observation_space_size],
            brain.camera_resolutions,
            policy_network_settings,
            ActionType.from_str(policy.act_type),
            self.act_size,
        )
        self.target_network = Critic(
            self.stream_names,
            policy_network_settings.hidden_units,
            [brain.vector_observation_space_size],
            brain.camera_resolutions,
            policy_network_settings.normalize,
            policy_network_settings.num_layers,
            policy_network_settings.memory.memory_size
            if policy_network_settings.memory is not None
            else 0,
            policy_network_settings.vis_encode_type,
        )
        self.soft_update(self.policy.actor_critic.critic, self.target_network, 1.0)

        self._log_ent_coef = torch.nn.Parameter(
            torch.log(torch.as_tensor([self.init_entcoef] * len(self.act_size))),
            requires_grad=True,
        )
        self.target_entropy = torch.as_tensor(
            -1
            * self.continuous_target_entropy_scale
            * np.prod(self.act_size[0]).astype(np.float32)
        )

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

        self.policy_optimizer = torch.optim.Adam(policy_params, lr=lr)
        self.value_optimizer = torch.optim.Adam(value_params, lr=lr)
        self.entropy_optimizer = torch.optim.Adam([self._log_ent_coef], lr=lr)

    def sac_q_loss(
        self,
        q1_out: Dict[str, torch.Tensor],
        q2_out: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        rewards: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        discrete: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates training-specific Tensorflow ops for SAC models.
        :param q1_streams: Q1 streams from policy network
        :param q1_streams: Q2 streams from policy network
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        :param stream_names: List of reward stream names.
        :param discrete: Whether or not to use discrete action losses.
        """
        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(q1_out.keys()):
            q1_stream = q1_out[name]
            q2_stream = q2_out[name]
            with torch.no_grad():
                q_backup = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[i]
                    * target_values[name]
                )
            _q1_loss = 0.5 * torch.mean(
                loss_masks * torch.pow((q_backup - q1_stream), 2)
            )
            _q2_loss = 0.5 * torch.mean(
                loss_masks * torch.pow((q_backup - q2_stream), 2)
            )

            q1_losses.append(_q1_loss)
            q2_losses.append(_q2_loss)
        q1_loss = torch.mean(torch.stack(q1_losses))
        q2_loss = torch.mean(torch.stack(q2_losses))
        return q1_loss, q2_loss

    def soft_update(self, source: nn.Module, target: nn.Module, tau: float) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

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

        for name in values.keys():
            if not discrete:
                min_policy_qs[name] = torch.min(q1p_out[name], q2p_out[name])
                _ent_coef = torch.exp(self._log_ent_coef)

        if not discrete:
            value_losses = []
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.sum(
                        _ent_coef * log_probs, dim=1
                    )
                value_loss = 0.5 * torch.mean(
                    loss_masks * torch.pow((values[name] - v_backup), 2)
                )
                value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def sac_policy_loss(
        self,
        log_probs: torch.Tensor,
        q1p_outs: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
        discrete: bool,
    ) -> torch.Tensor:
        _ent_coef = torch.exp(self._log_ent_coef)
        if not discrete:
            mean_q1 = torch.mean(torch.stack(list(q1p_outs.values())), axis=0)
            mean_q1.unsqueeze_(1)
            batch_policy_loss = torch.mean(_ent_coef * log_probs - mean_q1, dim=1)
            policy_loss = torch.mean(loss_masks * batch_policy_loss)
        else:
            policy_loss = 0
        return policy_loss

    def sac_entropy_loss(
        self, log_probs: torch.Tensor, loss_masks: torch.Tensor, discrete: bool
    ) -> torch.Tensor:
        if not discrete:
            with torch.no_grad():
                inner_term = torch.sum(log_probs + self.target_entropy, dim=1)
            entropy_loss = -torch.mean(self._log_ent_coef * loss_masks * inner_term)
        else:
            entropy_loss = 0
        return entropy_loss

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
            rewards[name] = torch.as_tensor(batch["{}_rewards".format(name)])

        vec_obs = [torch.as_tensor(batch["vector_obs"])]
        next_vec_obs = [torch.as_tensor(batch["next_vector_in"])]
        act_masks = torch.as_tensor(batch["action_mask"])
        if self.policy.use_continuous_act:
            actions = torch.as_tensor(batch["actions"]).unsqueeze(-1)
        else:
            actions = torch.as_tensor(batch["actions"], dtype=torch.long)

        memories = [
            torch.as_tensor(batch["memory"][i])
            for i in range(0, len(batch["memory"]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        vis_obs: List[torch.Tensor] = []
        next_vis_obs: List[torch.Tensor] = []
        if self.policy.use_vis_obs:
            vis_obs = []
            for idx, _ in enumerate(
                self.policy.actor_critic.network_body.visual_encoders
            ):
                vis_ob = torch.as_tensor(batch["visual_obs%d" % idx])
                vis_obs.append(vis_ob)
                next_vis_ob = torch.as_tensor(batch["next_visual_obs%d" % idx])
                next_vis_obs.append(next_vis_ob)

        # Copy normalizers from policy
        self.value_network.q1_network.copy_normalization(
            self.policy.actor_critic.network_body
        )
        self.value_network.q2_network.copy_normalization(
            self.policy.actor_critic.network_body
        )
        self.target_network.network_body.copy_normalization(
            self.policy.actor_critic.network_body
        )

        sampled_actions, log_probs, entropies, sampled_values, _ = self.policy.sample_actions(
            vec_obs,
            vis_obs,
            masks=act_masks,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        q1p_out, q2p_out = self.value_network(vec_obs, vis_obs, sampled_actions)
        q1_out, q2_out = self.value_network(vec_obs, vis_obs, actions.squeeze(-1))

        target_values, _ = self.target_network(next_vec_obs, next_vis_obs)
        q1_loss, q2_loss = self.sac_q_loss(
            q1_out,
            q2_out,
            target_values,
            torch.as_tensor(batch["done"]),
            rewards,
            torch.as_tensor(batch["masks"], dtype=torch.int32),
            False,
        )
        value_loss = self.sac_value_loss(
            log_probs,
            sampled_values,
            q1p_out,
            q2p_out,
            torch.as_tensor(batch["masks"], dtype=torch.int32),
            False,
        )
        policy_loss = self.sac_policy_loss(
            log_probs,
            q1p_out,
            torch.as_tensor(batch["masks"], dtype=torch.int32),
            False,
        )
        entropy_loss = self.sac_entropy_loss(
            log_probs, torch.as_tensor(batch["masks"], dtype=torch.int32), False
        )
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        total_value_loss = q1_loss + q2_loss + value_loss
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # Update Q network
        self.soft_update(self.policy.actor_critic.critic, self.target_network, self.tau)

        update_stats = {
            "Losses/Policy Loss": abs(policy_loss.detach().numpy()),
            "Losses/Value Loss": value_loss.detach().numpy(),
            "Losses/Q1 Loss": q1_loss.detach().numpy(),
            "Losses/Q2 Loss": q2_loss.detach().numpy(),
            "Policy/Entropy Coeff": torch.exp(self._log_ent_coef).detach().numpy(),
        }
        return update_stats

    def update_reward_signals(
        self, reward_signal_minibatches: Mapping[str, AgentBuffer], num_sequences: int
    ) -> Dict[str, float]:
        return {}

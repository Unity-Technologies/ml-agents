import copy
from typing import Dict, List, Tuple

import numpy as np

from mlagents.torch_utils import nn, torch, default_device
from mlagents.trainers.buffer import AgentBuffer, AgentBufferField
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.exception import TrainerConfigError, TrainerError
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import ASESettings
from mlagents.trainers.torch_entities.action_flattener import ActionFlattener
from mlagents.trainers.torch_entities.components.reward_providers import (
    BaseRewardProvider,
)
from mlagents.trainers.torch_entities.layers import linear_layer
from mlagents.trainers.torch_entities.networks import NetworkBody
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.base_env import BehaviorSpec, ObservationSpec


class ASERewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings: ASESettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = False
        self._discriminator_encoder = DiscriminatorEncoder(specs, settings)
        _, self._demo_buffer = demo_to_buffer(settings.demo_path, 1, specs, True)
        self._settings = settings
        params = list(self._discriminator_encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=settings.learning_rate)
        self.diversity_objective_weight = settings.omega_do
        self._update_batch_size = settings.batch_size
        self._beta_sdo = self._settings.beta_sdo
        self._encoder_scaling = self._settings.encoder_scaling

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            disc_reward, encoder_reward = self._discriminator_encoder.compute_rewards(
                mini_batch
            )

        return ModelUtils.to_numpy(
            disc_reward.squeeze(dim=1)
            + encoder_reward.squeeze(dim=1) * self._settings.beta_sdo
        )

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        expert_batch = self._demo_buffer.sample_mini_batch(self._update_batch_size, 1)

        if self._update_batch_size > mini_batch.num_experiences:
            raise TrainerError(
                "Discriminator batch size should be less than Policy batch size."
            )

        if self._update_batch_size <= mini_batch.num_experiences:
            mini_batch = mini_batch.sample_mini_batch(self._update_batch_size)

        self._discriminator_encoder.discriminator_network_body.update_normalization_with_next(
            expert_batch
        )

        (
            disc_loss,
            disc_stats_dict,
        ) = self._discriminator_encoder.compute_discriminator_loss(
            mini_batch, expert_batch
        )
        enc_loss, enc_stats_dict = self._discriminator_encoder.compute_encoder_loss(
            mini_batch
        )
        loss = disc_loss + self._settings.encoder_scaling * enc_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        stats_dict = {**disc_stats_dict, **enc_stats_dict}
        return stats_dict

    def compute_diversity_loss(
        self, policy: TorchPolicy, policy_mus: torch.Tensor, policy_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        return self._discriminator_encoder.compute_diversity_loss(
            policy, policy_mus, policy_batch
        )


class DiscriminatorEncoder(nn.Module):
    EPSILON = 1e-7

    def __init__(
        self,
        behavior_spec: BehaviorSpec,
        ase_settings: ASESettings,
    ):
        super().__init__()
        # need to modify obs specs to separate sending the latents into the disc/encoder -
        # observation_specs = behavior_spec.observation_specs
        observation_specs = copy.deepcopy(behavior_spec.observation_specs)
        self.latent_key = self._get_latent_key(observation_specs)
        del observation_specs[self.latent_key]
        for idx in range(len(observation_specs)):
            observation_specs[idx] = observation_specs[idx]._replace(
                shape=(2 * observation_specs[idx].shape[0],)
            )
        network_settings = ase_settings.network_settings
        self.discriminator_network_body = NetworkBody(
            observation_specs, network_settings
        )
        if ase_settings.shared_discriminator:
            self.encoder_network_body = self.discriminator_network_body
        else:
            self.encoder_network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.bottleneck_last:
            self.encoding_size = network_settings.hidden_units // 2
        else:
            self.encoding_size = network_settings.hidden_units

        self.discriminator_output_layer = nn.Sequential(
            linear_layer(self.encoding_size, 1, kernel_gain=0.2),
            nn.Sigmoid(),
        )
        # self.discriminator_output_layer = nn.Sequential(
        #     linear_layer(self.encoding_size, 1, kernel_gain=0.2)
        # )

        self.encoder_output_layer = nn.Linear(
            self.encoding_size, ase_settings.latent_dim
        )
        self.latent_dim = ase_settings.latent_dim
        self.encoder_reward_scale = ase_settings.encoder_scaling
        self.discriminator_reward_scale = 1
        self.gradient_penalty_weight = ase_settings.omega_gp
        self.weight_decay = ase_settings.omega_wd
        self.dics_logit_reg = ase_settings.disc_logit_reg
        self._action_flattener = ActionFlattener(behavior_spec.action_spec)

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        discriminator_network_output, _ = self.discriminator_network_body(inputs)
        encoder_network_output, _ = self.encoder_network_body(inputs)
        discriminator_output = self.discriminator_output_layer(
            discriminator_network_output
        )
        encoder_output = self.encoder_output_layer(encoder_network_output)
        encoder_output = torch.nn.functional.normalize(encoder_output, dim=-1)
        return discriminator_output, encoder_output

    def compute_rewards(
        self, mini_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # self.discriminator_network_body.update_normalization(mini_batch)
        disc_output, enc_output = self._compute_estimates(mini_batch)
        ase_latents = self._get_ase_latents(mini_batch)
        enc_reward = self._calc_encoder_reward(enc_output, ase_latents)
        disc_reward = self._calc_disc_reward(disc_output)
        return disc_reward, enc_reward

    def compute_discriminator_loss(
        self, policy_batch: AgentBuffer, expert_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        # needs to compute the loss like ase:amp_agent.py:470, includes samples from a replay buffer???
        # uses torch.nn.bcewithlogitloss, so need to remove sigmoid at the output of the disc
        # also need to change gradient mag computation
        total_loss = torch.zeros(1)
        stats_dict: Dict[str, np.ndarray] = {}
        policy_estimate, _ = self._compute_estimates(policy_batch)
        expert_estimate, _ = self._compute_estimates_expert(expert_batch)

        # stats_dict[
        #     "Policy/ASE Discriminator Policy Estimate"
        # ] = torch.sigmoid(policy_estimate).mean().item()
        # stats_dict[
        #     "Policy/ASE Discriminator Expert Estimate"
        # ] = torch.sigmoid(expert_estimate).mean().item()

        stats_dict[
            "Policy/ASE Discriminator Policy Estimate"
        ] = policy_estimate.mean().item()
        stats_dict[
            "Policy/ASE Discriminator Expert Estimate"
        ] = expert_estimate.mean().item()

        discriminator_loss = -(
            torch.log(expert_estimate + self.EPSILON)
            + torch.log(1.0 - policy_estimate + self.EPSILON)
        ).mean()
        total_loss += discriminator_loss

        # policy_loss = self._calc_disc_loss_neg(policy_estimate)
        # expert_loss = self._calc_disc_loss_pos(expert_estimate)
        # discriminator_loss = 0.5 * (expert_loss + policy_loss)

        total_loss += discriminator_loss

        # discriminator logit regularization
        if self.dics_logit_reg > 0:
            disc_logit_loss = self.dics_logit_reg * self._compute_disc_logit_loss()
            total_loss += disc_logit_loss

        # weight decay regularization
        if self.weight_decay > 0:
            weight_decay_loss = self.weight_decay * self._compute_weight_decay()
            stats_dict["Losses/ASE Weight Decay Loss"] = weight_decay_loss.item()
            total_loss += weight_decay_loss

        # grad penalty
        if self.gradient_penalty_weight > 0:
            gradient_magnitude_loss = (
                self.gradient_penalty_weight
                * self._compute_gradient_magnitude(policy_batch, expert_batch)
            )
            stats_dict["Losses/ASE Grad Mag Loss"] = gradient_magnitude_loss.item()
            total_loss += gradient_magnitude_loss

        stats_dict["Losses/ASE Discriminator Loss"] = total_loss.item()

        return total_loss, stats_dict

    def compute_encoder_loss(
        self, policy_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        total_loss = torch.zeros(1)
        stats_dict: Dict[str, np.ndarray] = {}
        _, encoder_prediction = self._compute_estimates(policy_batch)
        ase_latents = self._get_ase_latents(policy_batch)
        error = self._calc_encoder_error(encoder_prediction, ase_latents)
        total_loss += torch.mean(error)
        stats_dict["Losses/ASE Encoder Loss"] = total_loss.item()
        return total_loss, stats_dict

    def compute_diversity_loss(
        self, policy: TorchPolicy, policy_mus: torch.Tensor, policy_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        # currently only supports continuous actions
        # TODO add exception if not solely continuous actions
        total_loss = torch.zeros(1)
        stats_dict: Dict[str, np.ndarray] = {}
        ase_latents = self._get_ase_latents(policy_batch)
        batch_size = policy_batch.num_experiences
        new_latents = self._sample_latents(batch_size)
        new_policy_batch = self._replace_latents(policy_batch, new_latents)
        new_policy_mus = self._get_mus(policy, new_policy_batch)
        # clipping here  just like we do when stepping the environment
        clipped_policy_mus = torch.clamp(policy_mus, -3.0, 3.0) / 3.0
        clipped_new_policy_mus = torch.clamp(new_policy_mus, -3.0, 3.0) / 3.0
        a_diff = clipped_policy_mus - clipped_new_policy_mus
        a_diff = torch.mean(torch.square(a_diff), dim=-1)
        new_latents = torch.as_tensor(new_latents, dtype=torch.float)
        z_diff = new_latents * ase_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff
        diversity_bonus = a_diff / (z_diff + 1e-5)
        total_loss += torch.mean(torch.square(1.0 - diversity_bonus))
        stats_dict["Losses/ASE Diversity Loss"] = total_loss.item()
        return total_loss, stats_dict

    def _replace_latents(self, mini_batch: AgentBuffer, new_latents: np.ndarray):  # type: ignore
        new_mini_batch = copy.deepcopy(mini_batch)
        new_latents = AgentBufferField(new_latents.tolist())
        new_mini_batch[ObsUtil.get_name_at(self.latent_key)] = new_latents
        return new_mini_batch

    def _get_state_inputs(
        self, mini_batch: AgentBuffer, ignore_latent: bool = True
    ) -> List[torch.Tensor]:
        n_obs = len(self.discriminator_network_body.processors) + 1
        np_obs = ObsUtil.from_buffer(mini_batch, n_obs)
        # Convert to tensors
        # tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]
        tensor_obs = []
        for index, obs in enumerate(np_obs):
            if ignore_latent and index == self.latent_key:
                continue
            tensor_obs.append(ModelUtils.list_to_tensor(obs))

        return tensor_obs

    def _get_next_state_inputs(
        self, mini_batch: AgentBuffer, ignore_latent: bool = True
    ) -> List[torch.Tensor]:
        n_obs = len(self.discriminator_network_body.processors) + 1
        np_obs_next = ObsUtil.from_buffer_next(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = []
        for index, obs in enumerate(np_obs_next):
            if ignore_latent and index == self.latent_key:
                continue
            tensor_obs.append(ModelUtils.list_to_tensor(obs))

        return tensor_obs

    def _get_state_inputs_expert(self, mini_batch: AgentBuffer):  # type: ignore
        n_obs = len(self.discriminator_network_body.processors)
        np_obs = ObsUtil.from_buffer(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]

        return tensor_obs

    def _get_next_state_inputs_expert(self, mini_batch: AgentBuffer):  # type: ignore
        n_obs = len(self.discriminator_network_body.processors)
        np_obs = ObsUtil.from_buffer_next(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]

        return tensor_obs

    def _get_mus(self, policy: TorchPolicy, mini_batch: AgentBuffer) -> torch.Tensor:
        obs = self._get_state_inputs(mini_batch, False)
        run_out = policy.actor.get_mus(obs)
        mus = run_out["mus"]
        return mus

    def _get_ase_latents(self, mini_batch: AgentBuffer) -> torch.Tensor:
        inputs = self._get_state_inputs(mini_batch, False)
        ase_latents = inputs[self.latent_key]
        return ase_latents

    def _compute_estimates(
        self, mini_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._get_state_inputs(mini_batch)
        next_inputs = self._get_next_state_inputs(mini_batch)
        inputs_cat = [
            torch.cat([inp, next_inp], dim=1)
            for inp, next_inp in zip(inputs, next_inputs)
        ]
        disc_output, enc_output = self.forward(inputs_cat)

        return disc_output, enc_output

    def _compute_estimates_expert(
        self, mini_batch: AgentBuffer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._get_state_inputs_expert(mini_batch)
        next_inputs = self._get_next_state_inputs_expert(mini_batch)
        inputs_cat = [
            torch.cat([inp, next_inp], dim=1)
            for inp, next_inp in zip(inputs, next_inputs)
        ]
        disc_output, enc_output = self.forward(inputs_cat)
        return disc_output, enc_output

    def _compute_gradient_magnitude(
        self, policy_batch: AgentBuffer, expert_batch: AgentBuffer
    ) -> torch.Tensor:
        policy_inputs = self._get_state_inputs(policy_batch)
        next_policy_inputs = self._get_next_state_inputs(policy_batch)
        expert_inputs = self._get_state_inputs_expert(expert_batch)
        next_expert_inputs = self._get_next_state_inputs_expert(expert_batch)

        cat_policy_inputs = [
            torch.cat([inp, next_inp], dim=1)
            for inp, next_inp in zip(policy_inputs, next_policy_inputs)
        ]
        cat_expert_inputs = [
            torch.cat([inp, next_inp], dim=1)
            for inp, next_inp in zip(expert_inputs, next_expert_inputs)
        ]
        interp_inputs = []
        for policy_input, expert_input in zip(cat_policy_inputs, cat_expert_inputs):
            obs_epsilon = torch.rand(policy_input.shape)
            interp_input = obs_epsilon * policy_input + (1 - obs_epsilon) * expert_input
            interp_input.requires_grad = True  # For gradient calculation
            interp_inputs.append(interp_input)
        hidden, _ = self.discriminator_network_body(interp_inputs)
        # discriminator_output = torch.sigmoid(self.discriminator_output_layer(hidden))
        discriminator_output = self.discriminator_output_layer(hidden)
        estimate = discriminator_output.squeeze(1).sum()
        gradient = torch.autograd.grad(
            estimate, tuple(interp_inputs), create_graph=True
        )[0]
        safe_norm = (torch.sum(gradient**2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

    def _compute_weight_decay(self):
        discriminator_weights = self._get_discriminator_weights()
        discriminator_weights = torch.cat(discriminator_weights, dim=-1)
        discriminator_weight_decay = torch.sum(torch.square(discriminator_weights))
        return discriminator_weight_decay

    def _compute_disc_logit_loss(self):
        discriminator_logit_weights = self._get_discriminator_logit_weights()
        loss = torch.sum(torch.square(discriminator_logit_weights))
        return loss

    def _sample_latents(self, n) -> np.ndarray:  # type: ignore
        # torch version for future reference
        # z = torch.normal(torch.zeros([n, self.latent_dim], device=default_device()))
        # z = torch.nn.functional.normalize(z, dim=-1)
        # return z
        z = np.random.normal(size=(n, self.latent_dim))
        denom = np.linalg.norm(z, axis=1, keepdims=True)
        return z / denom

    def _calc_encoder_reward(
        self, encoder_prediction: torch.Tensor, ase_latents: torch.Tensor
    ) -> torch.Tensor:
        error = self._calc_encoder_error(encoder_prediction, ase_latents)
        enc_reward = torch.clamp(-error, 0.0)
        enc_reward *= self.encoder_reward_scale
        return enc_reward

    def _calc_disc_reward(self, discriminator_prediction: torch.Tensor) -> torch.Tensor:
        # probs = torch.sigmoid(discriminator_prediction)
        probs = discriminator_prediction
        disc_reward = -torch.log(
            torch.maximum(
                1 - probs,
                torch.tensor(0.0001, device=default_device()),
            )
        )
        disc_reward *= self.discriminator_reward_scale
        return disc_reward

    def _get_discriminator_weights(self):
        weights = []
        for module in self.discriminator_network_body.modules():
            if isinstance(module, nn.Linear):
                weights.append(torch.flatten(module.weight))
        for module in self.discriminator_output_layer.modules():
            if isinstance(module, nn.Linear):
                weights.append(torch.flatten(module.weight))
        return weights

    def _get_discriminator_logit_weights(self):
        weights = []
        for module in self.discriminator_output_layer.modules():
            if isinstance(module, nn.Linear):
                weights.append(torch.flatten(module.weight))
        return weights[0]

    @staticmethod
    def _calc_disc_loss_neg(discriminator_estimate: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(discriminator_estimate, torch.zeros_like(discriminator_estimate))
        return loss

    @staticmethod
    def _calc_disc_loss_pos(discriminator_estimate: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(discriminator_estimate, torch.ones_like(discriminator_estimate))
        return loss

    @staticmethod
    def _calc_encoder_error(
        encoder_prediction: torch.Tensor, ase_latents: torch.Tensor
    ) -> torch.Tensor:
        return -torch.sum(encoder_prediction * ase_latents, dim=-1, keepdim=True)

    @staticmethod
    def _get_latent_key(observation_specs: List[ObservationSpec]) -> int:  # type: ignore
        try:
            for idx, spec in enumerate(observation_specs):
                if spec.name == "EmbeddingSensor":
                    return idx
        except KeyError:
            raise TrainerConfigError("Something's wrong")

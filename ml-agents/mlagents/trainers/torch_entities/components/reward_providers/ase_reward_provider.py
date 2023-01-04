from typing import Dict, List, Tuple

import numpy as np

from mlagents.torch_utils import nn, torch, default_device
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.settings import ASESettings, NetworkSettings
from mlagents.trainers.torch_entities.components.reward_providers import BaseRewardProvider
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
        self._settings = settings

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            disc_reward, encoder_reward = self._discriminator_encoder.compute_estimates(mini_batch)

        return ModelUtils.to_numpy(
            disc_reward.squeeze(dim=1) + encoder_reward.squeeze(dim=1) * self._settings.beta_sdo)

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        pass


class DiscriminatorEncoder(nn.Module):
    def __init__(
        self,
        behavior_spec: BehaviorSpec,
        ase_settings: ASESettings,
    ):
        super().__init__()
        observation_specs = behavior_spec.observation_specs
        network_settings = ase_settings.network_settings
        self.discriminator_network_body = NetworkBody(
            observation_specs, network_settings
        )
        if ase_settings.shared_discriminator:
            self.encoder_network_body = self.discriminator_network_body
        else:
            self.encoder_network_body = NetworkBody(observation_specs, network_settings)
        self.encoding_size = network_settings.hidden_units
        self.discriminator_output_layer = nn.Sequential(linear_layer(network_settings.hidden_units, 1, kernel_gain=0.2), nn.Sigmoid())
        # self.discriminator_output_layer = nn.Sequential(linear_layer(network_settings.hidden_units, 1, kernel_gain=0.2))

        self.encoder_output_layer = nn.Linear(self.encoding_size, ase_settings.latent_dim)
        self.latent_key = self.get_latent_key(observation_specs)
        self.encoder_reward_scale = ase_settings.encoder_scaling
        self.discriminator_reward_scale = 1

    @staticmethod
    def get_latent_key(observation_specs: List[ObservationSpec]) -> int:
        try:
            for idx, spec in enumerate(observation_specs):
                if spec.name == "EmbeddingSensor":
                    return idx
        except KeyError:
            raise TrainerConfigError("Something's wrong")

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        discriminator_network_output, _ = self.discriminator_network_body(inputs)
        encoder_network_output, _ = self.encoder_network_body(inputs)
        return self.discriminator_output_layer(discriminator_network_output), self.encoder_output_layer(
            encoder_network_output)

    def get_state_inputs(self, mini_batch: AgentBuffer) -> List[torch.Tensor]:
        n_obs = len(self.discriminator_network_body.processors)
        np_obs = ObsUtil.from_buffer(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]
        return tensor_obs

    def compute_estimates(self, mini_batch: AgentBuffer) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.get_state_inputs(mini_batch)
        disc_output, enc_output = self.forward(inputs)
        ase_latents = inputs[self.latent_key]
        enc_reward = self._calc_encoder_reward(enc_output, ase_latents)
        disc_reward = self._calc_disc_reward(disc_output)
        return disc_reward, enc_reward

    def _calc_encoder_reward(self, encoder_prediction: torch.Tensor, ase_latents: torch.Tensor) -> torch.Tensor:
        error = self._calc_encoder_error(encoder_prediction, ase_latents)
        enc_reward = torch.clamp(-error, 0.0)
        enc_reward *= self.encoder_reward_scale
        return enc_reward

    def _calc_disc_reward(self, discriminator_prediction: torch.Tensor) -> torch.Tensor:
        # prob = 1 / (1 + torch.exp(-discriminator_prediction))
        disc_reward = -torch.log(torch.maximum(1 - discriminator_prediction, torch.tensor(0.0001, device=default_device())))
        disc_reward *= self.discriminator_reward_scale
        return disc_reward

    @staticmethod
    def _calc_encoder_error(encoder_prediction: torch.Tensor, ase_latents: torch.Tensor) -> torch.Tensor:
        return -torch.sum(encoder_prediction * ase_latents, dim=-1, keepdim=True)

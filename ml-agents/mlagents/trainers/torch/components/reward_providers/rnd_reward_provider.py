import numpy as np
from typing import Dict
import torch

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import RNDSettings

from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.networks import NetworkBody
from mlagents.trainers.torch.encoders import Normalizer
from mlagents.trainers.settings import NetworkSettings, EncoderType


class RNDRewardProvider(BaseRewardProvider):
    """
    https://arxiv.org/pdf/1810.12894.pdf
    """

    def __init__(self, specs: BehaviorSpec, settings: RNDSettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = True
        self._random_network = RNDNetwork(specs, settings)
        self._training_network = RNDNetwork(specs, settings)
        self._reward_normalizer = Normalizer(1)
        self.optimizer = torch.optim.Adam(
            self._training_network.parameters(), lr=settings.learning_rate
        )
        self._has_updated_once = False

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            target = self._random_network(mini_batch)
            prediction = self._training_network(mini_batch)
            unnormalized_rewards = torch.sum((prediction - target) ** 2, dim=1)
            rewards = self._reward_normalizer(unnormalized_rewards)
        self._reward_normalizer.update(unnormalized_rewards)
        return rewards.detach().cpu().numpy() * self._has_updated_once

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        self._has_updated_once = True
        with torch.no_grad():
            target = self._random_network(mini_batch)
        prediction = self._training_network(mini_batch)
        loss = torch.mean(torch.sum((prediction - target) ** 2, dim=1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"Losses/RND Loss": loss.detach().cpu().numpy()}


class RNDNetwork(torch.nn.Module):
    EPSILON = 1e-10

    def __init__(self, specs: BehaviorSpec, settings: RNDSettings) -> None:
        super().__init__()
        self._policy_specs = specs
        state_encoder_settings = NetworkSettings(
            normalize=True,
            hidden_units=settings.encoding_size,
            num_layers=3,
            vis_encode_type=EncoderType.SIMPLE,
            memory=None,
        )
        self._encoder = NetworkBody(specs.observation_shapes, state_encoder_settings)

    def forward(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the current state embedding from a mini_batch.
        """
        n_vis = len(self._encoder.visual_encoders)
        hidden, _ = self._encoder.forward(
            vec_inputs=[
                ModelUtils.list_to_tensor(mini_batch["vector_obs"], dtype=torch.float)
            ],
            vis_inputs=[
                ModelUtils.list_to_tensor(
                    mini_batch["visual_obs%d" % i], dtype=torch.float
                )
                for i in range(n_vis)
            ],
        )
        self._encoder.update_normalization(torch.tensor(mini_batch["vector_obs"]))
        return hidden

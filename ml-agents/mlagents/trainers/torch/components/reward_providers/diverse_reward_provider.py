import numpy as np
from typing import Dict
from mlagents.torch_utils import torch

from mlagents_envs.base_env import ObservationType
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import DiverseSettings
from mlagents.trainers.torch.action_flattener import ActionFlattener
from mlagents.trainers.torch.agent_action import AgentAction

from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs import logging_util
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.networks import NetworkBody
from mlagents.trainers.trajectory import ObsUtil

logger = logging_util.get_logger(__name__)


class DiverseRewardProvider(BaseRewardProvider):
    # From https://arxiv.org/pdf/1802.06070.pdf
    def __init__(self, specs: BehaviorSpec, settings: DiverseSettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = False  # Tried with false. Bias for staying alive.
        self._use_actions = False

        self._network = DiverseNetwork(specs, settings, self._use_actions)
        self.optimizer = torch.optim.SGD(
            self._network.parameters(), lr=settings.learning_rate
        )
        self._diverse_index = -1
        self._max_index = len(specs.observation_specs)
        for i, spec in enumerate(specs.observation_specs):
            if spec.observation_type == ObservationType.GOAL_SIGNAL:
                self._diverse_index = i

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            prediction = self._network(mini_batch)
            truth = ModelUtils.list_to_tensor(
                ObsUtil.from_buffer(mini_batch, self._max_index)[self._diverse_index]
            )
            # print(prediction[0,:], truth[0,:], torch.log(torch.sum((prediction * truth), dim=1) + 1e-10)[0], (torch.log(torch.sum((prediction * truth), dim=1))- np.log(1 / self._network.diverse_size))[0])
            rewards = torch.log(
                torch.sum((prediction * truth), dim=1) + 1e-10
            ) - np.log(1 / self._network.diverse_size)

        return rewards.detach().cpu().numpy()

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        all_loss = 0
        for _ in range(1):
            prediction = self._network(mini_batch)
            truth = ModelUtils.list_to_tensor(
                ObsUtil.from_buffer(mini_batch, self._max_index)[self._diverse_index]
            )
            # loss = torch.mean(
            #     torch.sum(-torch.log(prediction + 1e-10) * truth, dim=1), dim=0
            # )
            loss = -torch.mean(torch.log(torch.sum((prediction * truth), dim=1)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss += loss.item()
        return {"Losses/DIVERSE Loss": all_loss}

    def get_modules(self):
        return {f"Module:{self.name}": self._network}


class DiverseNetwork(torch.nn.Module):
    EPSILON = 1e-10

    def __init__(
        self, specs: BehaviorSpec, settings: DiverseSettings, use_actions: bool
    ) -> None:
        super().__init__()
        self._use_actions = use_actions
        state_encoder_settings = settings.network_settings
        if state_encoder_settings.memory is not None:
            state_encoder_settings.memory = None
            logger.warning(
                "memory was specified in network_settings but is not supported. It is being ignored."
            )
        self._action_flattener = ActionFlattener(specs.action_spec)
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

        print(" > ", new_spec, "\n\n\n", " >> ", diverse_spec)
        self._all_obs_specs = specs.observation_specs

        self.diverse_size = diverse_spec.shape[0]

        if self._use_actions:
            self._encoder = NetworkBody(
                new_spec, state_encoder_settings, self._action_flattener.flattened_size
            )
        else:
            self._encoder = NetworkBody(new_spec, state_encoder_settings)
        self._last_layer = torch.nn.Linear(
            state_encoder_settings.hidden_units, self.diverse_size
        )

    def forward(self, mini_batch: AgentBuffer) -> torch.Tensor:
        n_obs = len(self._encoder.processors) + 1
        np_obs = ObsUtil.from_buffer_next(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = [
            ModelUtils.list_to_tensor(obs)
            for obs, spec in zip(np_obs, self._all_obs_specs)
            if spec.observation_type != ObservationType.GOAL_SIGNAL
        ]

        if self._use_actions:
            action = self._action_flattener.forward(AgentAction.from_buffer(mini_batch))
            hidden, _ = self._encoder.forward(tensor_obs, action)
        else:
            hidden, _ = self._encoder.forward(tensor_obs)
        self._encoder.update_normalization(mini_batch)

        prediction = torch.softmax(self._last_layer(hidden), dim=1)
        return prediction

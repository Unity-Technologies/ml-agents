import numpy as np
from typing import Dict, NamedTuple
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import CuriositySettings

from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_flattener import ActionFlattener
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.networks import NetworkBody
from mlagents.trainers.torch.layers import LinearEncoder, linear_layer
from mlagents.trainers.settings import NetworkSettings, EncoderType


class ActionPredictionTuple(NamedTuple):
    continuous: torch.Tensor
    discrete: torch.Tensor


class CuriosityRewardProvider(BaseRewardProvider):
    beta = 0.2  # Forward vs Inverse loss weight
    loss_multiplier = 10.0  # Loss multiplier

    def __init__(self, specs: BehaviorSpec, settings: CuriositySettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = True
        self._network = CuriosityNetwork(specs, settings)
        self._network.to(default_device())

        self.optimizer = torch.optim.Adam(
            self._network.parameters(), lr=settings.learning_rate
        )
        self._has_updated_once = False

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            rewards = ModelUtils.to_numpy(self._network.compute_reward(mini_batch))
        rewards = np.minimum(rewards, 1.0 / self.strength)
        return rewards * self._has_updated_once

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        self._has_updated_once = True
        forward_loss = self._network.compute_forward_loss(mini_batch)
        inverse_loss = self._network.compute_inverse_loss(mini_batch)

        loss = self.loss_multiplier * (
            self.beta * forward_loss + (1.0 - self.beta) * inverse_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Losses/Curiosity Forward Loss": forward_loss.item(),
            "Losses/Curiosity Inverse Loss": inverse_loss.item(),
        }

    def get_modules(self):
        return {f"Module:{self.name}": self._network}


class CuriosityNetwork(torch.nn.Module):
    EPSILON = 1e-10

    def __init__(self, specs: BehaviorSpec, settings: CuriositySettings) -> None:
        super().__init__()
        self._action_spec = specs.action_spec
        state_encoder_settings = NetworkSettings(
            normalize=False,
            hidden_units=settings.encoding_size,
            num_layers=2,
            vis_encode_type=EncoderType.SIMPLE,
            memory=None,
        )
        self._state_encoder = NetworkBody(
            specs.observation_shapes, state_encoder_settings
        )

        self._action_flattener = ActionFlattener(self._action_spec)

        self.inverse_model_action_encoding = torch.nn.Sequential(
            LinearEncoder(2 * settings.encoding_size, 1, 256)
        )

        if self._action_spec.continuous_size > 0:
            self.continuous_action_prediction = linear_layer(
                256, self._action_spec.continuous_size
            )
        if self._action_spec.discrete_size > 0:
            self.discrete_action_prediction = linear_layer(
                256, sum(self._action_spec.discrete_branches)
            )

        self.forward_model_next_state_prediction = torch.nn.Sequential(
            LinearEncoder(
                settings.encoding_size + self._action_flattener.flattened_size, 1, 256
            ),
            linear_layer(256, settings.encoding_size),
        )

    def get_current_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the current state embedding from a mini_batch.
        """
        n_vis = len(self._state_encoder.visual_processors)
        hidden, _ = self._state_encoder.forward(
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
        return hidden

    def get_next_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the next state embedding from a mini_batch.
        """
        n_vis = len(self._state_encoder.visual_processors)
        hidden, _ = self._state_encoder.forward(
            vec_inputs=[
                ModelUtils.list_to_tensor(
                    mini_batch["next_vector_in"], dtype=torch.float
                )
            ],
            vis_inputs=[
                ModelUtils.list_to_tensor(
                    mini_batch["next_visual_obs%d" % i], dtype=torch.float
                )
                for i in range(n_vis)
            ],
        )
        return hidden

    def predict_action(self, mini_batch: AgentBuffer) -> ActionPredictionTuple:
        """
        In the continuous case, returns the predicted action.
        In the discrete case, returns the logits.
        """
        inverse_model_input = torch.cat(
            (self.get_current_state(mini_batch), self.get_next_state(mini_batch)), dim=1
        )

        continuous_pred = None
        discrete_pred = None
        hidden = self.inverse_model_action_encoding(inverse_model_input)
        if self._action_spec.continuous_size > 0:
            continuous_pred = self.continuous_action_prediction(hidden)
        if self._action_spec.discrete_size > 0:
            raw_discrete_pred = self.discrete_action_prediction(hidden)
            branches = ModelUtils.break_into_branches(
                raw_discrete_pred, self._action_spec.discrete_branches
            )
            branches = [torch.softmax(b, dim=1) for b in branches]
            discrete_pred = torch.cat(branches, dim=1)
        return ActionPredictionTuple(continuous_pred, discrete_pred)

    def predict_next_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Uses the current state embedding and the action of the mini_batch to predict
        the next state embedding.
        """
        actions = AgentAction.from_dict(mini_batch)
        flattened_action = self._action_flattener.forward(actions)
        forward_model_input = torch.cat(
            (self.get_current_state(mini_batch), flattened_action), dim=1
        )

        return self.forward_model_next_state_prediction(forward_model_input)

    def compute_inverse_loss(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Computes the inverse loss for a mini_batch. Corresponds to the error on the
        action prediction (given the current and next state).
        """
        predicted_action = self.predict_action(mini_batch)
        actions = AgentAction.from_dict(mini_batch)
        _inverse_loss = 0
        if self._action_spec.continuous_size > 0:
            sq_difference = (
                actions.continuous_tensor - predicted_action.continuous
            ) ** 2
            sq_difference = torch.sum(sq_difference, dim=1)
            _inverse_loss += torch.mean(
                ModelUtils.dynamic_partition(
                    sq_difference,
                    ModelUtils.list_to_tensor(mini_batch["masks"], dtype=torch.float),
                    2,
                )[1]
            )
        if self._action_spec.discrete_size > 0:
            true_action = torch.cat(
                ModelUtils.actions_to_onehot(
                    actions.discrete_tensor, self._action_spec.discrete_branches
                ),
                dim=1,
            )
            cross_entropy = torch.sum(
                -torch.log(predicted_action.discrete + self.EPSILON) * true_action,
                dim=1,
            )
            _inverse_loss += torch.mean(
                ModelUtils.dynamic_partition(
                    cross_entropy,
                    ModelUtils.list_to_tensor(
                        mini_batch["masks"], dtype=torch.float
                    ),  # use masks not action_masks
                    2,
                )[1]
            )
        return _inverse_loss

    def compute_reward(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Calculates the curiosity reward for the mini_batch. Corresponds to the error
        between the predicted and actual next state.
        """
        predicted_next_state = self.predict_next_state(mini_batch)
        target = self.get_next_state(mini_batch)
        sq_difference = 0.5 * (target - predicted_next_state) ** 2
        sq_difference = torch.sum(sq_difference, dim=1)
        return sq_difference

    def compute_forward_loss(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Computes the loss for the next state prediction
        """
        return torch.mean(
            ModelUtils.dynamic_partition(
                self.compute_reward(mini_batch),
                ModelUtils.list_to_tensor(mini_batch["masks"], dtype=torch.float),
                2,
            )[1]
        )

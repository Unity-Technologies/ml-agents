import numpy as np
import torch

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.reward_providers.base_reward_provider import BaseRewardProvider
from mlagents.trainers.settings import CuriositySettings

from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.networks import NetworkBody
from mlagents.trainers.settings import NetworkSettings, EncoderType


class CuriosityRewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings: CuriositySettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = True
        self._network = CuriosityNetwork(specs, settings)
        self.optimizer = torch.optim.Adam(
            self._network.parameters(), lr=settings.learning_rate
        )

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            rewards = self._network.compute_reward(mini_batch)
            return rewards.detach().cpu().numpy()

    def update(self, mini_batch: AgentBuffer) -> None:
        loss = self._network.compute_losses(mini_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CuriosityNetwork(torch.nn.Module):
    EPSILON = 1e-10
    forward_loss_weight = 2.0
    inverse_loss_weight = 8.0

    def __init__(self, specs: BehaviorSpec, settings: CuriositySettings) -> None:
        super().__init__()
        self._policy_specs = specs
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

        self._action_flattener = ModelUtils.ActionFlattener(specs)

        self.inverse_model_action_predition = torch.nn.Linear(
            2 * settings.encoding_size, self._action_flattener.flattened_size
        )

        self.forward_model_next_state_prediction = torch.nn.Linear(
            settings.encoding_size + self._action_flattener.flattened_size,
            settings.encoding_size,
        )

    def get_current_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the current state embedding from a mini_batch.
        """
        n_vis = len(self._state_encoder.visual_encoders)
        hidden, _ = self._state_encoder.forward(
            vec_inputs=[torch.as_tensor(mini_batch["vector_obs"], dtype=torch.float)],
            vis_inputs=[
                torch.as_tensor(mini_batch["visual_obs%d" % i], dtype=torch.float)
                for i in range(n_vis)
            ],
        )
        return hidden

    def get_next_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the next state embedding from a mini_batch.
        """
        n_vis = len(self._state_encoder.visual_encoders)
        hidden, _ = self._state_encoder.forward(
            vec_inputs=[
                torch.as_tensor(mini_batch["next_vector_in"], dtype=torch.float)
            ],
            vis_inputs=[
                torch.as_tensor(mini_batch["next_visual_obs%d" % i], dtype=torch.float)
                for i in range(n_vis)
            ],
        )
        return hidden

    def predict_action(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        In the continuous case, returns the predicted action.
        In the discrete case, returns the logits.
        """
        inverse_model_input = torch.cat(
            (self.get_current_state(mini_batch), self.get_next_state(mini_batch)), dim=1
        )
        inverse_model_input = ModelUtils.swish(inverse_model_input)
        hidden = self.inverse_model_action_predition(inverse_model_input)
        if self._policy_specs.is_action_continuous():
            return hidden
        else:
            branches = ModelUtils.break_into_branches(
                hidden, self._policy_specs.discrete_action_branches
            )
            branches = [torch.softmax(b, dim=1) for b in branches]
            return torch.cat(branches, dim=1)

    def predict_next_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Uses the current state embedding and the action of the mini_batch to predict
        the next state embedding.
        """
        if self._policy_specs.is_action_continuous():
            action = torch.as_tensor(mini_batch["actions"], dtype=torch.float)
        else:
            action = torch.cat(
                ModelUtils.actions_to_onehot(
                    torch.as_tensor(mini_batch["actions"], dtype=torch.long),
                    self._policy_specs.discrete_action_branches,
                ),
                dim=1,
            )
        forward_model_input = torch.cat(
            (self.get_current_state(mini_batch), action), dim=1
        )
        forward_model_input = ModelUtils.swish(forward_model_input)
        return self.forward_model_next_state_prediction(forward_model_input)

    def compute_inverse_loss(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Computes the inverse loss for a mini_batch. Corresponds to the error on the
        action prediction (given the current and next state).
        """
        predicted_action = self.predict_action(mini_batch)
        if self._policy_specs.is_action_continuous():
            sq_difference = (
                torch.as_tensor(mini_batch["actions"], dtype=torch.float)
                - predicted_action
            ) ** 2
            sq_difference = torch.sum(sq_difference, dim=1)
            return torch.mean(sq_difference)
        else:
            true_action = torch.cat(
                ModelUtils.actions_to_onehot(
                    torch.as_tensor(mini_batch["actions"], dtype=torch.long),
                    self._policy_specs.discrete_action_branches,
                ),
                dim=1,
            )
            cross_entropy = torch.sum(
                -torch.log(predicted_action + self.EPSILON) * true_action, dim=1
            )
            return torch.mean(
                ModelUtils.dynamic_partition(
                    cross_entropy,
                    torch.as_tensor(mini_batch["action_mask"], dtype=torch.float),
                    2,
                )[1]
            )

    def compute_reward(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Calculates the curiosity reward for the mini_batch. Corresponds to the error
        between the predicted and actual next state.
        """
        predicted_next_state = self.predict_next_state(mini_batch)
        sq_difference = (self.get_next_state(mini_batch) - predicted_next_state) ** 2
        sq_difference = torch.sum(sq_difference, dim=1)
        return sq_difference

    def compute_forward_loss(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Computes the loss for the next state prediction
        """
        return torch.mean(self.compute_reward(mini_batch))

    def compute_losses(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Computes the weighted sum of inverse and forward loss.
        """
        return self.forward_loss_weight * self.compute_forward_loss(
            mini_batch
        ) + self.inverse_loss_weight * self.compute_inverse_loss(mini_batch)

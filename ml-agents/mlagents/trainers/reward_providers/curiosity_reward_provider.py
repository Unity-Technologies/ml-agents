from typing import List
import numpy as np
import torch

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.reward_providers.base_reward_provider import BaseRewardProvider
from mlagents.trainers.settings import CuriositySettings

from mlagents_envs.base_env import BehaviorSpec


def swish(x):
    """
    TODO : MOVE SOMEWHERE BETTER
    """
    return x * torch.sigmoid(x)


class Swish(torch.nn.Module):
    """
    TODO : MOVE SOMEWHERE BETTER
    """

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mul(data, torch.sigmoid(data))


def actions_to_onehot(
    discrete_actions: torch.Tensor, action_size: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Splits a discrete action Tensor (of integers) into its one hot representations.
    Returns a list of Tensors (One Tensor per branch)
    :param discrete_actions: A Tensor of discrete actions.
    :param action_size: List of ints containing the number of possible actions for each branch.
    :return: A list of one hot Tensors (one or each branch).
    """
    onehot_branches = [
        torch.nn.functional.one_hot(_act.T, action_size[i]).float()
        for i, _act in enumerate(discrete_actions.long().T)
    ]
    return onehot_branches


def break_into_branches(
    concatenated_logits: torch.Tensor, action_size: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Takes a concatenated set of logits that represent multiple discrete action branches
    and breaks it up into one Tensor per branch.
    :param concatenated_logits: Tensor that represents the concatenated action branches
    :param action_size: List of ints containing the number of possible actions for each branch.
    :return: A List of Tensors containing one tensor per branch.
    """
    action_idx = [0] + list(np.cumsum(action_size))
    branched_logits = [
        concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
        for i in range(len(action_size))
    ]
    return branched_logits


def dynamic_partition(
    data: torch.Tensor, partitions: torch.Tensor, num_partitions: int
) -> List[torch.Tensor]:
    """
    Torch implementation of dynamic_partition :
    https://www.tensorflow.org/api_docs/python/tf/dynamic_partition
    Splits the data Tensor input into num_partitions Tensors according to the indices in
    partitions.
    :param data: The Tensor data that will be split into partitions.
    :param partitions: An indices tensor that determines in which partition each element
    of data will be in.
    :param num_partitions: The number of partitions to output. Corresponds to the
    maximum possible index in the partitions argument.
    :return: A list of Tensor partitions (Their indices correspond to their partition index).
    """
    res: List[torch.Tensor] = []
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res


class CuriosityRewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings: CuriositySettings) -> None:
        super().__init__(specs, settings)
        self._network = CuriosityNetwork(specs, settings)
        params = list(self._network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=settings.learning_rate)

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
        vec_obs_size = sum(
            shape[0] for shape in specs.observation_shapes if len(shape) == 1
        )
        # vis_obs_shapes = [shape for shape in specs.observation_shapes if len(shape) == 3]
        self._policy_specs = specs

        obs_size = vec_obs_size  # Only vector for now
        if obs_size > 0:
            self.vec_encode_1 = torch.nn.Linear(obs_size, settings.encoding_size)
            self.vec_encode_last = torch.nn.Linear(
                settings.encoding_size, settings.encoding_size
            )
        # TODO : The vector obs (Use networkBody from models_torch.py)
        if self._policy_specs.is_action_continuous():
            self.inverse_model_action_predition = torch.nn.Linear(
                2 * settings.encoding_size, self._policy_specs.action_size
            )
            self.forward_model_next_state_prediction = torch.nn.Linear(
                settings.encoding_size + self._policy_specs.action_size,
                settings.encoding_size,
            )
        else:
            self.inverse_model_action_predition = torch.nn.Linear(
                2 * settings.encoding_size,
                sum(self._policy_specs.discrete_action_branches),
            )
            self.forward_model_next_state_prediction = torch.nn.Linear(
                settings.encoding_size
                + sum(self._policy_specs.discrete_action_branches),
                settings.encoding_size,
            )

    def get_current_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the current state embedding from a mini_batch.
        """
        hidden = self.vec_encode_1(
            torch.as_tensor(mini_batch["vector_obs"], dtype=torch.float)
        )
        # TODO do visual
        hidden = swish(hidden)
        hidden = self.vec_encode_last(hidden)
        return hidden

    def get_next_state(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Extracts the next state embedding from a mini_batch.
        """
        hidden = self.vec_encode_1(
            torch.as_tensor(mini_batch["next_vector_in"], dtype=torch.float)
        )
        # TODO do visual
        hidden = swish(hidden)
        hidden = self.vec_encode_last(hidden)
        return hidden

    def predict_action(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        In the continuous case, returns the predicted action.
        In the discrete case, returns the logits.
        """
        inverse_model_input = torch.cat(
            (self.get_current_state(mini_batch), self.get_next_state(mini_batch)), dim=1
        )
        inverse_model_input = swish(inverse_model_input)
        hidden = self.inverse_model_action_predition(inverse_model_input)
        if self._policy_specs.is_action_continuous():
            return hidden
        else:
            branches = break_into_branches(
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
                actions_to_onehot(
                    torch.as_tensor(mini_batch["actions"], dtype=torch.long),
                    self._policy_specs.discrete_action_branches,
                ),
                dim=1,
            )
        forward_model_input = torch.cat(
            (self.get_current_state(mini_batch), action), dim=1
        )
        forward_model_input = swish(forward_model_input)
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
                actions_to_onehot(
                    torch.as_tensor(mini_batch["actions"], dtype=torch.long),
                    self._policy_specs.discrete_action_branches,
                ),
                dim=1,
            )
            cross_entropy = torch.sum(
                -torch.log(predicted_action + self.EPSILON) * true_action, dim=1
            )
            return torch.mean(
                dynamic_partition(
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

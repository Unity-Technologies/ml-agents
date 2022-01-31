from typing import Optional, Dict, List
import numpy as np
import random
from collections import defaultdict

from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import GAILSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs import logging_util
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_flattener import ActionFlattener
from mlagents.trainers.torch.networks import NetworkBody
from mlagents.trainers.torch.layers import linear_layer, Initialization, LayerNorm
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.trajectory import ObsUtil

logger = logging_util.get_logger(__name__)


class PreferenceRewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings) -> None:
        super().__init__(specs, settings)
        self._reward_model= RewardModel(specs, settings)
        self._reward_model.to(default_device())
        #_, self._demo_buffer = demo_to_buffer(
        #    settings.demo_path, 1, specs
        #)  # This is supposed to be the sequence length but we do not have access here
        params = list(self._reward_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=settings.learning_rate, weight_decay=1e-5)
        self.trajectory_and_rewards: List[List[AgentBuffer, float]] = []
        self.estimates: List[torch.Tensor] = []
        self._stats = defaultdict(list)

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        #self.trajectory_list.append(mini_batch)
        #self.rewards.append(np.sum(mini_batch[BufferKey.ENVIRONMENT_REWARDS]))
        self.trajectory_and_rewards.append([mini_batch, np.sum(mini_batch[BufferKey.ENVIRONMENT_REWARDS])])
        self._reward_model.encoder.update_normalization(mini_batch)
        
        if len(self.trajectory_and_rewards) > 1:
            for _ in range(3):
                self._train()
        with torch.no_grad():
            rew = self._reward_model.compute_rewards(
                mini_batch
            )
            rew_arr = ModelUtils.to_numpy(rew) 
            return rew_arr

            #normalized_rew = 0.05 * (rew_arr - rew_arr.mean()) / (rew_arr.std() + 1e-10)
            #return normalized_rew

    def _train(self):
        num_trajectories = 2 #min(2, len(self.trajectory_list))
        traj_rewards = random.sample(self.trajectory_and_rewards, num_trajectories)
        loss = torch.zeros(1)
        t1, r1 = traj_rewards[0]
        t1_est = torch.exp(torch.sum(self._reward_model.compute_rewards(t1)) / 100)
        t2, r2 = traj_rewards[1]
        t2_est = torch.exp(torch.sum(self._reward_model.compute_rewards(t2)) / 100)

        ratio = .5
        if r1 > r2:
            ratio = 1
        elif r1 < r2:
            ratio = 0
        #ratio = r1 / (r1 + r2)
        reward_loss, stats = self._reward_model.compute_loss(t1_est, t2_est, ratio)

        accuracy = 0
        if t1_est.item() > t2_est.item() and ratio > .5:
            accuracy = 1
        elif t1_est.item() < t2_est.item() and ratio < .5:
            accuracy = 1
        elif t1_est.item() == t2_est.item() and ratio == .5:
            accuracy = 1
        self._stats["Losses/RewardModel Loss"].append(reward_loss.item())
        self._stats["Reward Model/Accuracy"].append(accuracy)
        self._stats["Policy/Reward Mean"].append(stats["Policy/Reward Mean"]) 
        self.optimizer.zero_grad()
        reward_loss.backward()
        self.optimizer.step()

    def update(self, batch) -> Dict[str, np.ndarray]:
        return self._stats

    def get_modules(self):
        return {f"Module:{self.name}": self._reward_model}

    def clear(self):
        self._stats = defaultdict(list)
        #self.trajectory_list.clear()
        #self.rewards.clear()

class RewardModel(torch.nn.Module):

    def __init__(self, specs: BehaviorSpec, settings) -> None:
        super().__init__()
        self._settings = settings

        encoder_settings = settings.network_settings
        
        self._action_flattener = ActionFlattener(specs.action_spec)

        self.encoder = NetworkBody(
            specs.observation_specs, encoder_settings, self._action_flattener.flattened_size
        )
        estimator_input_size = encoder_settings.hidden_units
        self._estimator = linear_layer(estimator_input_size, 1, kernel_gain=0.2)

    def get_action_input(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Creates the action Tensor. In continuous case, corresponds to the action. In
        the discrete case, corresponds to the concatenation of one hot action Tensors.
        """
        return self._action_flattener.forward(AgentAction.from_buffer(mini_batch))

    def get_state_inputs(self, mini_batch: AgentBuffer) -> List[torch.Tensor]:
        """
        Creates the observation input.
        """
        n_obs = len(self.encoder.processors)
        np_obs = ObsUtil.from_buffer(mini_batch, n_obs)
        # Convert to tensors
        tensor_obs = [ModelUtils.list_to_tensor(obs) for obs in np_obs]
        return tensor_obs

    def compute_rewards(
        self, mini_batch: AgentBuffer
    ) -> torch.Tensor:
        inputs = self.get_state_inputs(mini_batch)
        actions = self.get_action_input(mini_batch)
        hidden, _ = self.encoder(inputs, actions)
        rewards = self._estimator(hidden).squeeze()
        return torch.tanh(rewards)
        #r_mean = torch.mean(rewards, dim=0, keepdim=True)
        #var = torch.mean((rewards - r_mean) ** 2, dim=0, keepdim=True)
        #norm_rew = (rewards - r_mean) / torch.sqrt(var + 1e-5)
        #return norm_rew

    def compute_loss(
        self, t1, t2, ratio
    ) -> torch.Tensor:
        """
        Given a policy mini_batch and an expert mini_batch, computes the loss of the discriminator.
        """
        
        stats_dict = {}
        stats_dict["Policy/Reward Mean"] = (t1).mean().item()
        t1_p = t1 / (t1 + t2)
        t2_p = t2 / (t1 + t2)
        reward_loss = - (ratio * torch.log(t1_p) + (1 - ratio) * torch.log(t2_p))
        return reward_loss, stats_dict

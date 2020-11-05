from typing import Optional, Dict, List, Tuple
import numpy as np
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.torch.components.reward_providers.base_reward_provider import (
    BaseRewardProvider,
)
from mlagents.trainers.settings import GAILSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.networks import NetworkBody
from mlagents.trainers.torch.layers import linear_layer, Initialization
from mlagents.trainers.settings import NetworkSettings, EncoderType
from mlagents.trainers.demo_loader import demo_to_buffer


class GAILRewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings: GAILSettings) -> None:
        super().__init__(specs, settings)
        self._ignore_done = True
        self._discriminator_network = DiscriminatorNetwork(specs, settings)
        self._discriminator_network.to(default_device())
        _, self._demo_buffer = demo_to_buffer(
            settings.demo_path, 1, specs
        )  # This is supposed to be the sequence length but we do not have access here
        params = list(self._discriminator_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=settings.learning_rate)

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        with torch.no_grad():
            estimates, _ = self._discriminator_network.compute_estimate(
                mini_batch, use_vail_noise=False
            )
            return ModelUtils.to_numpy(
                -torch.log(
                    1.0
                    - estimates.squeeze(dim=1)
                    * (1.0 - self._discriminator_network.EPSILON)
                )
            )

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        expert_batch = self._demo_buffer.sample_mini_batch(
            mini_batch.num_experiences, 1
        )
        loss, stats_dict = self._discriminator_network.compute_loss(
            mini_batch, expert_batch
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return stats_dict

    def get_modules(self):
        return {f"Module:{self.name}": self._discriminator_network}


class DiscriminatorNetwork(torch.nn.Module):
    gradient_penalty_weight = 10.0
    z_size = 128
    alpha = 0.0005
    mutual_information = 0.5
    EPSILON = 1e-7
    initial_beta = 0.0

    def __init__(self, specs: BehaviorSpec, settings: GAILSettings) -> None:
        super().__init__()
        self._use_vail = settings.use_vail
        self._settings = settings

        encoder_settings = NetworkSettings(
            normalize=False,
            hidden_units=settings.encoding_size,
            num_layers=2,
            vis_encode_type=EncoderType.SIMPLE,
            memory=None,
        )
        self._action_flattener = ModelUtils.ActionFlattener(specs.action_spec)
        unencoded_size = (
            self._action_flattener.flattened_size + 1 if settings.use_actions else 0
        )  # +1 is for dones
        self.encoder = NetworkBody(
            specs.observation_shapes, encoder_settings, unencoded_size
        )

        estimator_input_size = settings.encoding_size
        if settings.use_vail:
            estimator_input_size = self.z_size
            self._z_sigma = torch.nn.Parameter(
                torch.ones((self.z_size), dtype=torch.float), requires_grad=True
            )
            self._z_mu_layer = linear_layer(
                settings.encoding_size,
                self.z_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=0.1,
            )
            self._beta = torch.nn.Parameter(
                torch.tensor(self.initial_beta, dtype=torch.float), requires_grad=False
            )

        self._estimator = torch.nn.Sequential(
            linear_layer(estimator_input_size, 1), torch.nn.Sigmoid()
        )

    def get_action_input(self, mini_batch: AgentBuffer) -> torch.Tensor:
        """
        Creates the action Tensor. In continuous case, corresponds to the action. In
        the discrete case, corresponds to the concatenation of one hot action Tensors.
        """
        return self._action_flattener.forward(
            torch.as_tensor(mini_batch["actions"], dtype=torch.float)
        )

    def get_state_inputs(
        self, mini_batch: AgentBuffer
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Creates the observation input.
        """
        n_vis = len(self.encoder.visual_processors)
        n_vec = len(self.encoder.vector_processors)
        vec_inputs = (
            [ModelUtils.list_to_tensor(mini_batch["vector_obs"], dtype=torch.float)]
            if n_vec > 0
            else []
        )
        vis_inputs = [
            ModelUtils.list_to_tensor(mini_batch["visual_obs%d" % i], dtype=torch.float)
            for i in range(n_vis)
        ]
        return vec_inputs, vis_inputs

    def compute_estimate(
        self, mini_batch: AgentBuffer, use_vail_noise: bool = False
    ) -> torch.Tensor:
        """
        Given a mini_batch, computes the estimate (How much the discriminator believes
        the data was sampled from the demonstration data).
        :param mini_batch: The AgentBuffer of data
        :param use_vail_noise: Only when using VAIL : If true, will sample the code, if
        false, will return the mean of the code.
        """
        vec_inputs, vis_inputs = self.get_state_inputs(mini_batch)
        if self._settings.use_actions:
            actions = self.get_action_input(mini_batch)
            dones = torch.as_tensor(mini_batch["done"], dtype=torch.float).unsqueeze(1)
            action_inputs = torch.cat([actions, dones], dim=1)
            hidden, _ = self.encoder(vec_inputs, vis_inputs, action_inputs)
        else:
            hidden, _ = self.encoder(vec_inputs, vis_inputs)
        z_mu: Optional[torch.Tensor] = None
        if self._settings.use_vail:
            z_mu = self._z_mu_layer(hidden)
            hidden = torch.normal(z_mu, self._z_sigma * use_vail_noise)
        estimate = self._estimator(hidden)
        return estimate, z_mu

    def compute_loss(
        self, policy_batch: AgentBuffer, expert_batch: AgentBuffer
    ) -> torch.Tensor:
        """
        Given a policy mini_batch and an expert mini_batch, computes the loss of the discriminator.
        """
        total_loss = torch.zeros(1)
        stats_dict: Dict[str, np.ndarray] = {}
        policy_estimate, policy_mu = self.compute_estimate(
            policy_batch, use_vail_noise=True
        )
        expert_estimate, expert_mu = self.compute_estimate(
            expert_batch, use_vail_noise=True
        )
        stats_dict["Policy/GAIL Policy Estimate"] = policy_estimate.mean().item()
        stats_dict["Policy/GAIL Expert Estimate"] = expert_estimate.mean().item()
        discriminator_loss = -(
            torch.log(expert_estimate + self.EPSILON)
            + torch.log(1.0 - policy_estimate + self.EPSILON)
        ).mean()
        stats_dict["Losses/GAIL Loss"] = discriminator_loss.item()
        total_loss += discriminator_loss
        if self._settings.use_vail:
            # KL divergence loss (encourage latent representation to be normal)
            kl_loss = torch.mean(
                -torch.sum(
                    1
                    + (self._z_sigma ** 2).log()
                    - 0.5 * expert_mu ** 2
                    - 0.5 * policy_mu ** 2
                    - (self._z_sigma ** 2),
                    dim=1,
                )
            )
            vail_loss = self._beta * (kl_loss - self.mutual_information)
            with torch.no_grad():
                self._beta.data = torch.max(
                    self._beta + self.alpha * (kl_loss - self.mutual_information),
                    torch.tensor(0.0),
                )
            total_loss += vail_loss
            stats_dict["Policy/GAIL Beta"] = self._beta.item()
            stats_dict["Losses/GAIL KL Loss"] = kl_loss.item()
        if self.gradient_penalty_weight > 0.0:
            gradient_magnitude_loss = (
                self.gradient_penalty_weight
                * self.compute_gradient_magnitude(policy_batch, expert_batch)
            )
            stats_dict["Policy/GAIL Grad Mag Loss"] = gradient_magnitude_loss.item()
            total_loss += gradient_magnitude_loss
        return total_loss, stats_dict

    def compute_gradient_magnitude(
        self, policy_batch: AgentBuffer, expert_batch: AgentBuffer
    ) -> torch.Tensor:
        """
        Gradient penalty from https://arxiv.org/pdf/1704.00028. Adds stability esp.
        for off-policy. Compute gradients w.r.t randomly interpolated input.
        """
        policy_vec_inputs, policy_vis_inputs = self.get_state_inputs(policy_batch)
        expert_vec_inputs, expert_vis_inputs = self.get_state_inputs(expert_batch)
        interp_vec_inputs = []
        for policy_vec_input, expert_vec_input in zip(
            policy_vec_inputs, expert_vec_inputs
        ):
            obs_epsilon = torch.rand(policy_vec_input.shape)
            interp_vec_input = (
                obs_epsilon * policy_vec_input + (1 - obs_epsilon) * expert_vec_input
            )
            interp_vec_input.requires_grad = True  # For gradient calculation
            interp_vec_inputs.append(interp_vec_input)
        interp_vis_inputs = []
        for policy_vis_input, expert_vis_input in zip(
            policy_vis_inputs, expert_vis_inputs
        ):
            obs_epsilon = torch.rand(policy_vis_input.shape)
            interp_vis_input = (
                obs_epsilon * policy_vis_input + (1 - obs_epsilon) * expert_vis_input
            )
            interp_vis_input.requires_grad = True  # For gradient calculation
            interp_vis_inputs.append(interp_vis_input)
        if self._settings.use_actions:
            policy_action = self.get_action_input(policy_batch)
            expert_action = self.get_action_input(expert_batch)
            action_epsilon = torch.rand(policy_action.shape)
            policy_dones = torch.as_tensor(
                policy_batch["done"], dtype=torch.float
            ).unsqueeze(1)
            expert_dones = torch.as_tensor(
                expert_batch["done"], dtype=torch.float
            ).unsqueeze(1)
            dones_epsilon = torch.rand(policy_dones.shape)
            action_inputs = torch.cat(
                [
                    action_epsilon * policy_action
                    + (1 - action_epsilon) * expert_action,
                    dones_epsilon * policy_dones + (1 - dones_epsilon) * expert_dones,
                ],
                dim=1,
            )
            action_inputs.requires_grad = True
            hidden, _ = self.encoder(
                interp_vec_inputs, interp_vis_inputs, action_inputs
            )
            encoder_input = tuple(
                interp_vec_inputs + interp_vis_inputs + [action_inputs]
            )
        else:
            hidden, _ = self.encoder(interp_vec_inputs, interp_vis_inputs)
            encoder_input = tuple(interp_vec_inputs + interp_vis_inputs)
        if self._settings.use_vail:
            use_vail_noise = True
            z_mu = self._z_mu_layer(hidden)
            hidden = torch.normal(z_mu, self._z_sigma * use_vail_noise)
        estimate = self._estimator(hidden).squeeze(1).sum()
        gradient = torch.autograd.grad(estimate, encoder_input, create_graph=True)[0]
        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient ** 2, dim=1) + self.EPSILON).sqrt()
        gradient_mag = torch.mean((safe_norm - 1) ** 2)
        return gradient_mag

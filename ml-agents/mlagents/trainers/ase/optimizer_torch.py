from typing import cast, Dict

import attr

from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    OnPolicyHyperparamSettings,
    ScheduleType,
    TrainerSettings,
)
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.buffer import AgentBuffer
from timers import timed
from mlagents.torch_utils import torch, default_device
from mlagents.trainers.torch_entities.utils import ModelUtils


@attr.s(auto_attribs=True)
class ASESettings(OnPolicyHyperparamSettings):
    latent_dim: int = 16
    beta: float = 5.0e-3
    epsilon: float = 0.2
    num_epoch: int = 3
    beta_sdo: float = 0.5
    omega_gp: float = 5
    omega_do: float = 0.01
    encoder_scaling: float = 1
    spu: int = 32768
    pv_mini_batch: int = 4096
    de_mini_batch: int = 1024
    gae_lambda: float = 0.95
    td_lambda: float = 0.95
    shared_critic: bool = False
    shared_discriminator: bool = True
    learning_rate_schedule: ScheduleType = ScheduleType.CONSTANT
    beta_schedule: ScheduleType = ScheduleType.CONSTANT
    epsilon_schedule: ScheduleType = ScheduleType.CONSTANT


class TorchASEOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        self.hyperparameters: ASESettings = cast(
            ASESettings, trainer_settings.hyperparameters
        )

        params = list(self.policy.actor.parameters())
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())
            params += list(self._critic.parameters())

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

    @property
    def critic(self):
        return self._critic

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

    # TODO move module update into TorchOptimizer for reward_provider
    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

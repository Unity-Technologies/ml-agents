from typing import Dict, Any, List

import numpy as np

from base_env import DecisionSteps
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.policy import Policy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import GlobalSteps
from mlagents.torch_utils import default_device
from trainers.action_info import ActionInfo

EPSILON = 1e-7


class ASEPolicy(Policy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor_cls: type,
        actor_kwargs: Dict[str, Any],
        discriminator_cls: type,
        discriminator_kwargs: Dict[str, Any],
        encoder_cls: type,
        encoder_kwargs: Dict[str, Any],
    ):
        super().__init__(seed, behavior_spec, network_settings)
        self.global_step = GlobalSteps()

        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.actor = actor_cls(
            observation_specs=self.behavior_spec.observation_specs,
            network_settings=network_settings,
            action_spec=behavior_spec.action_spec,
            **actor_kwargs,
        )

        self.discriminator = discriminator_cls(
            observation_specs=self.behavior_spec.observation_specs,
            network_settings=network_settings,
            **discriminator_kwargs,
        )

        self.encoder = encoder_cls(**encoder_kwargs)

        # Save the m_size needed for export
        self._export_m_size = self.m_size
        # m_size needed for training is determined by network, not trainer settings
        self.m_size = self.actor.memory_size

        self.actor.to(default_device())
        self.discriminator.to(default_device())

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        pass

    def increment_step(self, n_steps):
        pass

    def get_current_step(self):
        pass

    def load_weights(self, values: List[np.ndarray]) -> None:
        pass

    def get_weights(self) -> List[np.ndarray]:
        pass

    def init_load_weights(self) -> None:
        pass

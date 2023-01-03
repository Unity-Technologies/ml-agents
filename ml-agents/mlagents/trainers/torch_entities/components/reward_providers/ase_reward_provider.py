from typing import Dict

import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.settings import ASESettings
from mlagents.trainers.torch_entities.components.reward_providers import BaseRewardProvider
from mlagents_envs.base_env import BehaviorSpec


class ASERewardProvider(BaseRewardProvider):
    def __init__(self, specs: BehaviorSpec, settings: ASESettings):
        super().__init__(specs, settings)

    def evaluate(self, mini_batch: AgentBuffer) -> np.ndarray:
        pass

    def update(self, mini_batch: AgentBuffer) -> Dict[str, np.ndarray]:
        pass

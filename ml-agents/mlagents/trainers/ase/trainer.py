# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (ASE)
# Contains an implementation of Adversarial Skill Embeddings as described in : https://arxiv.org/abs/2205.01906
from typing import cast
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.ase.optimizer_torch import TorchASEOptimizer

TRAINER_NAME = "ase"


class ASETrainer(PPOTrainer):
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

    def create_optimizer(self) -> TorchOptimizer:
        return TorchASEOptimizer(
            cast(TorchPolicy, self.policy), self.trainer_settings
        )

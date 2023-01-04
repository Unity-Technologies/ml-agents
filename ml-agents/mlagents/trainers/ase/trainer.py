# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (ASE)
# Contains an implementation of Adversarial Skill Embeddings as described in : https://arxiv.org/abs/2205.01906

from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.settings import TrainerSettings

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

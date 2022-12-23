# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (ASE)
# Contains an implementation of Adversarial Skill Embeddings as described in : https://arxiv.org/abs/2205.01906

from typing import cast, Type, Union, Dict, Any

# import numpy as np

from base_env import BehaviorSpec
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.ase.optimizer_torch import ASESettings
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.policy.ase_policy import ASEPolicy
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.networks import (
    SimpleActor,
    SharedActorCritic,
    SimpleDiscriminator,
    SkillEncoder,
)

TRAINER_NAME = "ase"


class ASETrainer(OnPolicyTrainer):
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

        self.hyperparameters: ASESettings = cast(
            ASESettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.shared_critic = self.hyperparameters.shared_critic
        self.policy: TorchPolicy = None  # type: ignore

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> ASEPolicy:
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        discriminator_cls: Type[SimpleDiscriminator] = SimpleDiscriminator
        encoder_cls: Type[SkillEncoder] = SkillEncoder
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }
        discriminator_kwargs: Dict[str, Any] = {}
        encoder_kwargs = {"skill_embedding_size": 64}

        if self.shared_critic:
            reward_signal_configs = self.trainer_settings.reward_signals
            reward_signal_names = [
                key.value for key, _ in reward_signal_configs.items()
            ]
            actor_cls = SharedActorCritic
            actor_kwargs.update({"stream_names": reward_signal_names})

        policy = ASEPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
            discriminator_cls,
            discriminator_kwargs,
            encoder_cls,
            encoder_kwargs,
        )

        return policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

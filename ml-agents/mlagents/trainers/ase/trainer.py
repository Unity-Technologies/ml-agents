# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (ASE)
# Contains an implementation of Adversarial Skill Embeddings as described in : https://arxiv.org/abs/2205.01906

from typing import cast, Type, Union, Dict, Any, Tuple

from mlagents.trainers.exception import TrainerConfigError
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.ase.optimizer_torch import ASESettings, TorchASEOptimizer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.policy.ase_policy import ASEPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.networks import (
    SimpleActor,
    SharedActorCritic,
    DiscriminatorEncoder,
)
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trajectory import Trajectory

# import numpy as np
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.policy import Policy

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
        self.shared_discriminator = self.hyperparameters.shared_discriminator
        self.policy: ASEPolicy = None  # type: ignore

    def _get_embedding_size_and_idx(self, behavior_spec: BehaviorSpec) -> Tuple[int, int]:
        for idx, spec in enumerate(behavior_spec.observation_specs):
            if spec.name == "EmbeddingSensor":
                return spec.shape[0], idx
        raise TrainerConfigError("For some reason, the agent is missing the embedding vector sensor!")

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> ASEPolicy:
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        disc_enc_cls: Type[DiscriminatorEncoder] = DiscriminatorEncoder
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }
        embedding_size, embedding_idx = self._get_embedding_size_and_idx(behavior_spec)
        disc_enc_kwargs: Dict[str, Any] = {"shared": self.shared_discriminator,
                                           "embedding_size": embedding_size}

        additional_kwargs: Dict[str, Any] = {"latent_steps_min": self.hyperparameters.latent_steps_min,
                                             "latent_steps_max": self.hyperparameters.latent_steps_max,
                                             "embedding_idx": embedding_idx}

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
            disc_enc_cls,
            disc_enc_kwargs,
            additional_kwargs
        )

        return policy

    def create_optimizer(self) -> TorchOptimizer:
        return TorchASEOptimizer(cast(TorchPolicy, self.policy), self.trainer_settings)

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        print(trajectory.behavior_id)

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

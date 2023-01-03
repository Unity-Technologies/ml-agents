# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (ASE)
# Contains an implementation of Adversarial Skill Embeddings as described in : https://arxiv.org/abs/2205.01906

from typing import cast, Type, Union, Dict, Any, Tuple
import numpy as np
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.ase.optimizer_torch import TorchASEOptimizer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.policy.ase_policy import ASEPolicy
from mlagents.trainers.settings import TrainerSettings, ASESettings, RewardSignalType
from mlagents.trainers.torch_entities.networks import (
    SimpleActor,
    SharedActorCritic,
    DiscriminatorEncoder,
)
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil

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

        self.hyperparameters: PPOSettings = cast(
            PPOSettings, self.trainer_settings.hyperparameters
        )
        self.ase_hyperparameters: ASESettings = cast(
            ASESettings, self.trainer_settings.reward_signals[RewardSignalType.ASE]
        )
        self.seed = seed
        self.shared_critic = self.hyperparameters.shared_critic
        self.shared_discriminator = self.ase_hyperparameters.shared_discriminator
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

        additional_kwargs: Dict[str, Any] = {"latent_steps_min": self.ase_hyperparameters.latent_steps_min,
                                             "latent_steps_max": self.ase_hyperparameters.latent_steps_max,
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
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id
        agent_buffer_trajectory = trajectory.to_agentbuffer()
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        (
            value_estimates,
            value_next,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                v
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )

        # Compute ASE reward
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )

        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Compute GAE and returns
        tmp_advantages = []
        tmp_returns = []
        for name in self.optimizer.reward_signals:
            bootstrap_value = value_next[name]

            local_rewards = agent_buffer_trajectory[
                RewardSignalUtil.rewards_key(name)
            ].get_batch()
            local_value_estimates = agent_buffer_trajectory[
                RewardSignalUtil.value_estimates_key(name)
            ].get_batch()

            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_value_estimates,
                value_next=bootstrap_value,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.gae_lambda,
            )
            local_return = local_advantage + local_value_estimates
            # This is later use as target for the different value estimates
            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
                local_return
            )
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
                local_advantage
            )
            tmp_advantages.append(local_advantage)
            tmp_returns.append(local_return)

        # Get global advantages
        global_advantages = list(
            np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
        )
        global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
        agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)

        self._append_to_update_buffer(agent_buffer_trajectory)

        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

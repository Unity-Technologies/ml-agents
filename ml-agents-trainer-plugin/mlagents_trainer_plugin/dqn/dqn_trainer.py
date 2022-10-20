from typing import cast

import numpy as np
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trajectory import Trajectory, ObsUtil
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import TrainerSettings
from .dqn_optimizer import DQNOptimizer, DQNSettings, QNetwork

logger = get_logger(__name__)
TRAINER_NAME = "dqn"


class DQNTrainer(OffPolicyTrainer):
    """The DQNTrainer is an implementation of"""

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
        """
        Responsible for collecting experiences and training SAC model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.policy: TorchPolicy = None  # type: ignore
        self.optimizer: DQNOptimizer = None  # type: ignore

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the replay buffer.
        """
        super()._process_trajectory(trajectory)
        last_step = trajectory.steps[-1]
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        # Check if we used group rewards, warn if so.
        self._warn_if_group_reward(agent_buffer_trajectory)

        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        # Evaluate all reward functions for reporting purposes
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )

            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Get all value estimates for reporting purposes
        (
            value_estimates,
            _,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory, trajectory.next_obs, trajectory.done_reached
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value",
                np.mean(v),
            )

        # Bootstrap using the last step rather than the bootstrap step if max step is reached.
        # Set last element to duplicate obs and remove dones.
        if last_step.interrupted:
            last_step_obs = last_step.obs
            for i, obs in enumerate(last_step_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)][-1] = obs
            agent_buffer_trajectory[BufferKey.DONE][-1] = False

        self._append_to_update_buffer(agent_buffer_trajectory)

        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_optimizer(self) -> TorchOptimizer:
        """
        Creates an Optimizer object
        """
        return DQNOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and give DQN hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        # initialize online Q-network which works as actor
        exploration_initial_eps = cast(
            DQNSettings, self.trainer_settings.hyperparameters
        ).exploration_initial_eps
        actor_kwargs = {
            "exploration_initial_eps": exploration_initial_eps,
            "stream_names": [
                signal.value for signal in self.trainer_settings.reward_signals.keys()
            ],
        }
        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls=QNetwork,
            actor_kwargs=actor_kwargs,
        )
        self.maybe_load_replay_buffer()
        return policy

    @staticmethod
    def get_settings_type():
        return DQNSettings

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME


def get_type_and_setting():
    return {DQNTrainer.get_trainer_name(): DQNTrainer}, {
        DQNTrainer.get_trainer_name(): DQNTrainer.get_settings_type()
    }

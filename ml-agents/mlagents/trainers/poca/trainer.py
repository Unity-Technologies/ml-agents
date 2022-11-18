# # Unity ML-Agents Toolkit
# ## ML-Agents Learning (POCA)
# Contains an implementation of MA-POCA.

from collections import defaultdict
from typing import cast, Dict, Union, Any, Type

import numpy as np

from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.trainer.trainer_utils import lambda_return
from mlagents.trainers.policy import Policy
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.poca.optimizer_torch import TorchPOCAOptimizer, POCASettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic

logger = get_logger(__name__)

TRAINER_NAME = "poca"


class POCATrainer(OnPolicyTrainer):
    """The POCATrainer is an implementation of the MA-POCA algorithm."""

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
        Responsible for collecting experiences and training POCA model.
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
        self.hyperparameters: POCASettings = cast(
            POCASettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.policy: TorchPolicy = None  # type: ignore
        self.optimizer: TorchPOCAOptimizer = None  # type: ignore
        self.collected_group_rewards: Dict[str, int] = defaultdict(lambda: 0)

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        :param trajectory: The Trajectory tuple containing the steps to be processed.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id  # All the agents should have the same ID

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        # Get all value estimates
        (
            value_estimates,
            baseline_estimates,
            value_next,
            value_memories,
            baseline_memories,
        ) = self.optimizer.get_trajectory_and_baseline_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.next_group_obs,
            trajectory.all_group_dones_reached
            and trajectory.done_reached
            and not trajectory.interrupted,
        )

        if value_memories is not None and baseline_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)
            agent_buffer_trajectory[BufferKey.BASELINE_MEMORY].set(baseline_memories)

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                v
            )
            agent_buffer_trajectory[
                RewardSignalUtil.baseline_estimates_key(name)
            ].extend(baseline_estimates[name])
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Baseline Estimate",
                np.mean(baseline_estimates[name]),
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(value_estimates[name]),
            )

        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        self.collected_group_rewards[agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.GROUP_REWARD]
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

        # Compute lambda returns and advantage
        tmp_advantages = []
        for name in self.optimizer.reward_signals:

            local_rewards = np.array(
                agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].get_batch(),
                dtype=np.float32,
            )

            baseline_estimate = agent_buffer_trajectory[
                RewardSignalUtil.baseline_estimates_key(name)
            ].get_batch()
            v_estimates = agent_buffer_trajectory[
                RewardSignalUtil.value_estimates_key(name)
            ].get_batch()

            lambd_returns = lambda_return(
                r=local_rewards,
                value_estimates=v_estimates,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
                value_next=value_next[name],
            )

            local_advantage = np.array(lambd_returns) - np.array(baseline_estimate)

            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
                lambd_returns
            )
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
                local_advantage
            )
            tmp_advantages.append(local_advantage)

        # Get global advantages
        global_advantages = list(
            np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
        )
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)

        self._append_to_update_buffer(agent_buffer_trajectory)

        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)
            # Remove dead agents from group reward recording
            if not trajectory.all_group_dones_reached:
                self.collected_group_rewards.pop(agent_id)

        # If the whole team is done, average the remaining group rewards.
        if trajectory.all_group_dones_reached and trajectory.done_reached:
            self.stats_reporter.add_stat(
                "Environment/Group Cumulative Reward",
                self.collected_group_rewards.get(agent_id, 0),
                aggregation=StatsAggregationMethod.HISTOGRAM,
            )
            self.collected_group_rewards.pop(agent_id)

    def _is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        size_of_buffer = self.update_buffer.num_experiences
        return size_of_buffer > self.hyperparameters.buffer_size

    def end_episode(self) -> None:
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets. For POCA, we should
        also zero out the group rewards.
        """
        super().end_episode()
        self.collected_group_rewards.clear()

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and POCA hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        return policy

    def create_optimizer(self) -> TorchPOCAOptimizer:
        return TorchPOCAOptimizer(self.policy, self.trainer_settings)

    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

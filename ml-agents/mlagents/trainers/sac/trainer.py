# ## ML-Agent Learning (SAC)
# Contains an implementation of SAC as described in https://arxiv.org/abs/1801.01290
# and implemented in https://github.com/hill-a/stable-baselines

from collections import defaultdict, deque
from typing import Dict, cast
import os

import numpy as np
from mlagents.trainers.policy.checkpoint_manager import ModelCheckpoint

from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import timed
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.base_env import ObservationType
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.policy import Policy
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer
from mlagents.trainers.trajectory import Trajectory, ObsUtil
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, SACSettings

logger = get_logger(__name__)

BUFFER_TRUNCATE_PERCENT = 0.8


class BehaviorMetrics:
    CONT_BINS = 20
    WINDOW_SIZE = 10

    def __init__(self, size, cont):
        self.cont = cont
        self.shape = (self.CONT_BINS,) * size if cont else (size,)
        num_metrics = self.CONT_BINS ** size if cont else size
        self._rewards = [deque([], self.WINDOW_SIZE) for _ in range(num_metrics)]

    def _behavior_to_idx(self, behavior):
        if self.cont:
            bins = list((behavior * self.CONT_BINS / 2 + self.CONT_BINS / 2).astype(int))
            idx = 0
            exp = 0
            while len(bins) > 0:
                idx += bins.pop() * self.CONT_BINS ** exp
                exp += 1
        else:
            idx = np.argmax(behavior)
        return idx

    def add_reward(self, reward, behavior):
        self._rewards[self._behavior_to_idx(behavior)].append(reward)

    def get_avg_reward_array(self):
        avg = np.array([np.mean(x) if len(x) > 0 else 0 for x in self._rewards])
        return np.reshape(avg, self.shape)

    def get_avg_reward(self, behavior):
        return np.mean(self._rewards[self._behavior_to_idx(behavior)])

    def get_last_reward(self, behavior):
        return self._rewards[self._behavior_to_idx(behavior)][-1]

    def is_full(self):
        return all(len(x) == self.WINDOW_SIZE for x in self._rewards)

    def percent_full(self):
        return sum(len(x) for x in self._rewards) / (np.prod(self.shape) * self.WINDOW_SIZE)


class SACTrainer(RLTrainer):
    """
    The SACTrainer is an implementation of the SAC algorithm, with support
    for discrete actions and recurrent networks.
    """

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
            trainer_settings,
            training,
            load,
            artifact_path,
            reward_buff_cap,
        )

        self.seed = seed
        self.policy: Policy = None  # type: ignore
        self.optimizer: TorchSACOptimizer = None  # type: ignore
        self.hyperparameters: SACSettings = cast(
            SACSettings, trainer_settings.hyperparameters
        )
        self._step = 0

        # Don't divide by zero
        self.update_steps = 1
        self.reward_signal_update_steps = 1

        self.steps_per_update = self.hyperparameters.steps_per_update
        self.reward_signal_steps_per_update = (
            self.hyperparameters.reward_signal_steps_per_update
        )

        self.checkpoint_replay_buffer = self.hyperparameters.save_replay_buffer

        self.behavior_reward_hist = dict()
        self.behavior_metrics = None
        self.log_steps = trainer_settings.summary_freq
        self.log_count = 0

    def _checkpoint(self) -> ModelCheckpoint:
        """
        Writes a checkpoint model to memory
        Overrides the default to save the replay buffer.
        """
        ckpt = super()._checkpoint()
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()
        return ckpt

    def save_model(self) -> None:
        """
        Saves the final training model to memory
        Overrides the default to save the replay buffer.
        """
        super().save_model()
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()

    def save_replay_buffer(self) -> None:
        """
        Save the training buffer's update buffer to a pickle file.
        """
        filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
        logger.info(f"Saving Experience Replay Buffer to {filename}...")
        with open(filename, "wb") as file_object:
            self.update_buffer.save_to_file(file_object)
            logger.info(
                f"Saved Experience Replay Buffer ({os.path.getsize(filename)} bytes)."
            )

    def load_replay_buffer(self) -> None:
        """
        Loads the last saved replay buffer from a file.
        """
        filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
        logger.info(f"Loading Experience Replay Buffer from {filename}...")
        with open(filename, "rb+") as file_object:
            self.update_buffer.load_from_file(file_object)
        logger.debug(
            "Experience replay buffer has {} experiences.".format(
                self.update_buffer.num_experiences
            )
        )

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
            self.policy.update_normalization(agent_buffer_trajectory)

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

            if self.hyperparameters.mede:
                diversity_vector = [
                    obs
                    for obs, spec in zip(trajectory.steps[0].obs, 
                                        self.policy.behavior_spec.observation_specs)
                    if spec.observation_type == ObservationType.GOAL_SIGNAL
                ][0]
                reward = self.collected_rewards["environment"].get(agent_id, 0)
                self._update_mede_stats(diversity_vector, reward)

            self._update_end_episode_stats(agent_id, self.optimizer)

    def _update_mede_stats(self, diversity_vector, reward):
        if self.behavior_metrics is None:
            self.behavior_metrics = BehaviorMetrics(diversity_vector.shape[0], self.hyperparameters.mede_continuous)

        self.behavior_metrics.add_reward(reward, diversity_vector)

        reward_bins = self.behavior_metrics.get_avg_reward_array()
        self._stats_reporter.add_stat(
            "Settings/Max Environment Reward",
            np.amax(reward_bins)
        )

        if not self.hyperparameters.mede_continuous:
            diversity_setting = np.argmax(diversity_vector)
            self._stats_reporter.add_stat(
                "Settings/Environment Reward {}".format(diversity_setting),
                reward
            )

        if self._step > self.log_steps * self.log_count:
            self.log_count += 1

            if self.hyperparameters.mede_continuous:
                for dim in range(len(reward_bins.shape)):
                    mean_dims = list(range(len(reward_bins.shape)))
                    mean_dims.remove(dim)
                    hist = [
                        bn / self.behavior_metrics.CONT_BINS - 1 
                        for bn, r in enumerate(np.mean(reward_bins, axis=tuple(mean_dims))) for _ in range(int(r * 100))
                    ]
                    if len(hist) > 0:
                        self._stats_reporter.write_hist(
                            "Settings/Environment Reward {}".format(dim), 
                            hist, 
                            self._step
                        )
                if len(self.behavior_metrics.shape) == 2:
                    self._stats_reporter.write_image(
                        "Settings/Reward Map", 
                        np.expand_dims(reward_bins, 0) / np.amax(reward_bins), 
                        self._step
                    )

            with open(os.path.join(self.artifact_path, "reward_bins.npy"), "wb") as f:
                np.save(f, reward_bins)
            if not self.is_training:
                print("Inference progress: {}".format(self.behavior_metrics.percent_full()))
                with open(os.path.join(self.artifact_path, "reward_bins_{}.npy".format(self._step)), "wb") as f:
                    np.save(f, reward_bins)
                if self.behavior_metrics.is_full():
                    exit()

    def _is_ready_update(self) -> bool:
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not _update_policy() can be run
        """
        return (
            self.update_buffer.num_experiences >= self.hyperparameters.batch_size
            and self._step >= self.hyperparameters.buffer_init_steps
        )

    @timed
    def _update_policy(self) -> bool:
        """
        Update the SAC policy and reward signals. The reward signal generators are updated using different mini batches.
        By default we imitate http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times.
        :return: Whether or not the policy was updated.
        """
        policy_was_updated = self._update_sac_policy()
        self._update_reward_signals()
        return policy_was_updated

    def maybe_load_replay_buffer(self):
        # Load the replay buffer if load
        if self.load and self.checkpoint_replay_buffer:
            try:
                self.load_replay_buffer()
            except (AttributeError, FileNotFoundError):
                logger.warning(
                    "Replay buffer was unable to load, starting from scratch."
                )
            logger.debug(
                "Loaded update buffer with {} sequences".format(
                    self.update_buffer.num_experiences
                )
            )

    def create_torch_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and SAC hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings,
            condition_sigma_on_obs=True,
            tanh_squash=True,
            separate_critic=True,
        )
        self.maybe_load_replay_buffer()
        return policy

    def _update_sac_policy(self) -> bool:
        """
        Uses update_buffer to update the policy. We sample the update_buffer and update
        until the steps_per_update ratio is met.
        """
        has_updated = False
        self.cumulative_returns_since_policy_update.clear()
        n_sequences = max(
            int(self.hyperparameters.batch_size / self.policy.sequence_length), 1
        )

        batch_update_stats: Dict[str, list] = defaultdict(list)
        while (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.update_steps > self.steps_per_update:
            logger.debug(f"Updating SAC policy at step {self._step}")
            buffer = self.update_buffer
            if self.update_buffer.num_experiences >= self.hyperparameters.batch_size:
                sampled_minibatch = buffer.sample_mini_batch(
                    self.hyperparameters.batch_size,
                    sequence_length=self.policy.sequence_length,
                )
                # Get rewards for each reward
                for name, signal in self.optimizer.reward_signals.items():
                    sampled_minibatch[RewardSignalUtil.rewards_key(name)] = (
                        signal.evaluate(sampled_minibatch) * signal.strength
                    )

                update_stats = self.optimizer.update(sampled_minibatch, n_sequences)
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)

                self.update_steps += 1

                for stat, stat_list in batch_update_stats.items():
                    self._stats_reporter.add_stat(stat, np.mean(stat_list))
                has_updated = True

            if self.optimizer.bc_module:
                update_stats = self.optimizer.bc_module.update()
                for stat, val in update_stats.items():
                    self._stats_reporter.add_stat(stat, val)

        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        if self.update_buffer.num_experiences > self.hyperparameters.buffer_size:
            self.update_buffer.truncate(
                int(self.hyperparameters.buffer_size * BUFFER_TRUNCATE_PERCENT)
            )
        return has_updated

    def _update_reward_signals(self) -> None:
        """
        Iterate through the reward signals and update them. Unlike in PPO,
        do it separate from the policy so that it can be done at a different
        interval.
        This function should only be used to simulate
        http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times. Normally, the reward signal
        and policy are updated in parallel.
        """
        buffer = self.update_buffer
        n_sequences = max(
            int(self.hyperparameters.batch_size / self.policy.sequence_length), 1
        )
        batch_update_stats: Dict[str, list] = defaultdict(list)
        while (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.reward_signal_update_steps > self.reward_signal_steps_per_update:
            # Get minibatches for reward signal update if needed
            reward_signal_minibatches = {}
            for name in self.optimizer.reward_signals.keys():
                logger.debug(f"Updating {name} at step {self._step}")
                if name != "extrinsic":
                    reward_signal_minibatches[name] = buffer.sample_mini_batch(
                        self.hyperparameters.batch_size,
                        sequence_length=self.policy.sequence_length,
                    )
            update_stats = self.optimizer.update_reward_signals(
                reward_signal_minibatches, n_sequences
            )
            for stat_name, value in update_stats.items():
                batch_update_stats[stat_name].append(value)
            self.reward_signal_update_steps += 1

            for stat, stat_list in batch_update_stats.items():
                self._stats_reporter.add_stat(stat, np.mean(stat_list))

    def create_sac_optimizer(self) -> TorchSACOptimizer:
        return TorchSACOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy
    ) -> None:
        """
        Adds policy to trainer.
        """
        if self.policy:
            logger.warning(
                "Your environment contains multiple teams, but {} doesn't support adversarial games. Enable self-play to \
                    train adversarial games.".format(
                    self.__class__.__name__
                )
            )
        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy
        self.optimizer = self.create_sac_optimizer()
        for _reward_signal in self.optimizer.reward_signals.keys():
            self.collected_rewards[_reward_signal] = defaultdict(lambda: 0)

        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()

        # Needed to resume loads properly
        self._step = policy.get_current_step()
        # Assume steps were updated at the correct ratio before
        self.update_steps = int(max(1, self._step / self.steps_per_update))
        self.reward_signal_update_steps = int(
            max(1, self._step / self.reward_signal_steps_per_update)
        )

    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """

        return self.policy

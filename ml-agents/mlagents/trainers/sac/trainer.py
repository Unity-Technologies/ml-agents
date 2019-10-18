# # Unity ML-Agents Toolkit
# ## ML-Agent Learning (SAC)
# Contains an implementation of SAC as described in https://arxiv.org/abs/1801.01290
# and implemented in https://github.com/hill-a/stable-baselines

import logging
from collections import defaultdict
from typing import List, Dict
import os

import numpy as np

from mlagents.envs.brain import AllBrainInfo
from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.envs.timers import timed
from mlagents.trainers.sac.policy import SACPolicy
from mlagents.trainers.rl_trainer import RLTrainer, AllRewardsOutput


LOGGER = logging.getLogger("mlagents.trainers")
BUFFER_TRUNCATE_PERCENT = 0.8


class SACTrainer(RLTrainer):
    """
    The SACTrainer is an implementation of the SAC algorithm, with support
    for discrete actions and recurrent networks.
    """

    def __init__(
        self, brain, reward_buff_cap, trainer_parameters, training, load, seed, run_id
    ):
        """
        Responsible for collecting experiences and training SAC model.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param run_id: The The identifier of the current run
        """
        super().__init__(brain, trainer_parameters, training, run_id, reward_buff_cap)
        self.param_keys = [
            "batch_size",
            "buffer_size",
            "buffer_init_steps",
            "hidden_units",
            "learning_rate",
            "init_entcoef",
            "max_steps",
            "normalize",
            "num_update",
            "num_layers",
            "time_horizon",
            "sequence_length",
            "summary_freq",
            "tau",
            "use_recurrent",
            "summary_path",
            "memory_size",
            "model_path",
            "reward_signals",
            "vis_encode_type",
        ]

        self.check_param_keys()

        self.step = 0
        self.train_interval = (
            trainer_parameters["train_interval"]
            if "train_interval" in trainer_parameters
            else 1
        )
        self.reward_signal_updates_per_train = (
            trainer_parameters["reward_signals"]["reward_signal_num_update"]
            if "reward_signal_num_update" in trainer_parameters["reward_signals"]
            else trainer_parameters["num_update"]
        )

        self.checkpoint_replay_buffer = (
            trainer_parameters["save_replay_buffer"]
            if "save_replay_buffer" in trainer_parameters
            else False
        )
        self.policy = SACPolicy(seed, brain, trainer_parameters, self.is_training, load)

        # Load the replay buffer if load
        if load and self.checkpoint_replay_buffer:
            try:
                self.load_replay_buffer()
            except (AttributeError, FileNotFoundError):
                LOGGER.warning(
                    "Replay buffer was unable to load, starting from scratch."
                )
            LOGGER.debug(
                "Loaded update buffer with {} sequences".format(
                    len(self.training_buffer.update_buffer["actions"])
                )
            )

        for _reward_signal in self.policy.reward_signals.keys():
            self.collected_rewards[_reward_signal] = {}

        self.episode_steps = {}

    def save_model(self) -> None:
        """
        Saves the model. Overrides the default save_model since we want to save
        the replay buffer as well.
        """
        self.policy.save_model(self.get_step)
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()

    def save_replay_buffer(self) -> None:
        """
        Save the training buffer's update buffer to a pickle file.
        """
        filename = os.path.join(self.policy.model_path, "last_replay_buffer.hdf5")
        LOGGER.info("Saving Experience Replay Buffer to {}".format(filename))
        with open(filename, "wb") as file_object:
            self.training_buffer.update_buffer.save_to_file(file_object)

    def load_replay_buffer(self) -> None:
        """
        Loads the last saved replay buffer from a file.
        """
        filename = os.path.join(self.policy.model_path, "last_replay_buffer.hdf5")
        LOGGER.info("Loading Experience Replay Buffer from {}".format(filename))
        with open(filename, "rb+") as file_object:
            self.training_buffer.update_buffer.load_from_file(file_object)
        LOGGER.info(
            "Experience replay buffer has {} experiences.".format(
                len(self.training_buffer.update_buffer["actions"])
            )
        )

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        """
        Takes the output of the last action and store it into the training buffer.
        """
        actions = take_action_outputs["action"]
        self.training_buffer[agent_id]["actions"].append(actions[agent_idx])

    def add_rewards_outputs(
        self,
        rewards_out: AllRewardsOutput,
        values: Dict[str, np.ndarray],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> None:
        """
        Takes the value output of the last action and store it into the training buffer.
        """
        self.training_buffer[agent_id]["environment_rewards"].append(
            rewards_out.environment[agent_next_idx]
        )

    def process_experiences(
        self, current_info: AllBrainInfo, new_info: AllBrainInfo
    ) -> None:
        """
        Checks agent histories for processing condition, and processes them as necessary.
        :param current_info: Dictionary of all current brains and corresponding BrainInfo.
        :param new_info: Dictionary of all next brains and corresponding BrainInfo.
        """
        info = new_info[self.brain_name]
        if self.is_training:
            self.policy.update_normalization(info.vector_observations)
        for l in range(len(info.agents)):
            agent_actions = self.training_buffer[info.agents[l]]["actions"]
            if (
                info.local_done[l]
                or len(agent_actions) >= self.trainer_parameters["time_horizon"]
            ) and len(agent_actions) > 0:
                agent_id = info.agents[l]

                # Bootstrap using last brain info. Set last element to duplicate obs and remove dones.
                if info.max_reached[l]:
                    bootstrapping_info = self.training_buffer[agent_id].last_brain_info
                    idx = bootstrapping_info.agents.index(agent_id)
                    for i, obs in enumerate(bootstrapping_info.visual_observations):
                        self.training_buffer[agent_id]["next_visual_obs%d" % i][
                            -1
                        ] = obs[idx]
                    if self.policy.use_vec_obs:
                        self.training_buffer[agent_id]["next_vector_in"][
                            -1
                        ] = bootstrapping_info.vector_observations[idx]
                    self.training_buffer[agent_id]["done"][-1] = False

                self.training_buffer.append_update_buffer(
                    agent_id,
                    batch_size=None,
                    training_length=self.policy.sequence_length,
                )

                self.training_buffer[agent_id].reset_agent()
                if info.local_done[l]:
                    self.stats["Environment/Episode Length"].append(
                        self.episode_steps.get(agent_id, 0)
                    )
                    self.episode_steps[agent_id] = 0
                    for name, rewards in self.collected_rewards.items():
                        if name == "environment":
                            self.cumulative_returns_since_policy_update.append(
                                rewards.get(agent_id, 0)
                            )
                            self.stats["Environment/Cumulative Reward"].append(
                                rewards.get(agent_id, 0)
                            )
                            self.reward_buffer.appendleft(rewards.get(agent_id, 0))
                            rewards[agent_id] = 0
                        else:
                            self.stats[
                                self.policy.reward_signals[name].stat_name
                            ].append(rewards.get(agent_id, 0))
                            rewards[agent_id] = 0

    def is_ready_update(self) -> bool:
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not update_model() can be run
        """
        return (
            len(self.training_buffer.update_buffer["actions"])
            >= self.trainer_parameters["batch_size"]
            and self.step >= self.trainer_parameters["buffer_init_steps"]
        )

    @timed
    def update_policy(self) -> None:
        """
        If train_interval is met, update the SAC policy given the current reward signals.
        If reward_signal_train_interval is met, update the reward signals from the buffer.
        """
        if self.step % self.train_interval == 0:
            self.trainer_metrics.start_policy_update_timer(
                number_experiences=len(self.training_buffer.update_buffer["actions"]),
                mean_return=float(np.mean(self.cumulative_returns_since_policy_update)),
            )
            self.update_sac_policy()
            self.update_reward_signals()
            self.trainer_metrics.end_policy_update()

    def update_sac_policy(self) -> None:
        """
        Uses demonstration_buffer to update the policy.
        The reward signal generators are updated using different mini batches.
        If we want to imitate http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times, then reward_signal_updates_per_train
        is greater than 1 and the reward signals are not updated in parallel.
        """

        self.cumulative_returns_since_policy_update: List[float] = []
        n_sequences = max(
            int(self.trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )

        num_updates = self.trainer_parameters["num_update"]
        batch_update_stats: Dict[str, list] = defaultdict(list)
        for _ in range(num_updates):
            LOGGER.debug("Updating SAC policy at step {}".format(self.step))
            buffer = self.training_buffer.update_buffer
            if (
                len(self.training_buffer.update_buffer["actions"])
                >= self.trainer_parameters["batch_size"]
            ):
                sampled_minibatch = buffer.sample_mini_batch(
                    self.trainer_parameters["batch_size"],
                    sequence_length=self.policy.sequence_length,
                )
                # Get rewards for each reward
                for name, signal in self.policy.reward_signals.items():
                    sampled_minibatch[
                        "{}_rewards".format(name)
                    ] = signal.evaluate_batch(sampled_minibatch).scaled_reward

                update_stats = self.policy.update(
                    sampled_minibatch, n_sequences, update_target=True
                )
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)

        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        if (
            len(self.training_buffer.update_buffer["actions"])
            > self.trainer_parameters["buffer_size"]
        ):
            self.training_buffer.truncate_update_buffer(
                int(self.trainer_parameters["buffer_size"] * BUFFER_TRUNCATE_PERCENT)
            )

        for stat, stat_list in batch_update_stats.items():
            self.stats[stat].append(np.mean(stat_list))

        if self.policy.bc_module:
            update_stats = self.policy.bc_module.update()
            for stat, val in update_stats.items():
                self.stats[stat].append(val)

    def update_reward_signals(self) -> None:
        """
        Iterate through the reward signals and update them. Unlike in PPO,
        do it separate from the policy so that it can be done at a different
        interval.
        This function should only be used to simulate
        http://arxiv.org/abs/1809.02925 and similar papers, where the policy is updated
        N times, then the reward signals are updated N times. Normally, the reward signal
        and policy are updated in parallel.
        """
        buffer = self.training_buffer.update_buffer
        num_updates = self.reward_signal_updates_per_train
        n_sequences = max(
            int(self.trainer_parameters["batch_size"] / self.policy.sequence_length), 1
        )
        batch_update_stats: Dict[str, list] = defaultdict(list)
        for _ in range(num_updates):
            # Get minibatches for reward signal update if needed
            reward_signal_minibatches = {}
            for name, signal in self.policy.reward_signals.items():
                LOGGER.debug("Updating {} at step {}".format(name, self.step))
                # Some signals don't need a minibatch to be sampled - so we don't!
                if signal.update_dict:
                    reward_signal_minibatches[name] = buffer.sample_mini_batch(
                        self.trainer_parameters["batch_size"],
                        sequence_length=self.policy.sequence_length,
                    )
            update_stats = self.policy.update_reward_signals(
                reward_signal_minibatches, n_sequences
            )
            for stat_name, value in update_stats.items():
                batch_update_stats[stat_name].append(value)
        for stat, stat_list in batch_update_stats.items():
            self.stats[stat].append(np.mean(stat_list))

# # Unity ML-Agents Toolkit
import logging
from typing import Dict, NamedTuple
from collections import defaultdict
import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trainer import Trainer, UnityTrainerException
from mlagents.trainers.components.reward_signals import RewardSignalResult
from mlagents.trainers import stats

LOGGER = logging.getLogger("mlagents.trainers")

RewardSignalResults = Dict[str, RewardSignalResult]


class AllRewardsOutput(NamedTuple):
    """
    This class stores all of the outputs of the reward signals,
    as well as the raw reward from the environment.
    """

    reward_signals: RewardSignalResults
    environment: np.ndarray


class RLTrainer(Trainer):
    """
    This class is the base class for trainers that use Reward Signals.
    Contains methods for adding BrainInfos to the Buffer.
    """

    def __init__(self, *args, **kwargs):
        super(RLTrainer, self).__init__(*args, **kwargs)
        # Make sure we have at least one reward_signal
        if not self.trainer_parameters["reward_signals"]:
            raise UnityTrainerException(
                "No reward signals were defined. At least one must be used with {}.".format(
                    self.__class__.__name__
                )
            )
        # collected_rewards is a dictionary from name of reward signal to a dictionary of agent_id to cumulative reward
        # used for reporting only. We always want to report the environment reward to Tensorboard, regardless
        # of what reward signals are actually present.
        self.collected_rewards = {"environment": defaultdict(lambda: 0)}
        self.update_buffer = AgentBuffer()
        self.episode_steps = defaultdict(lambda: 0)

    def end_episode(self) -> None:
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        for agent_id in self.episode_steps:
            self.episode_steps[agent_id] = 0
        for rewards in self.collected_rewards.values():
            for agent_id in rewards:
                rewards[agent_id] = 0

    def _update_end_episode_stats(self, agent_id: str) -> None:
        self.episode_steps[agent_id] = 0
        for name, rewards in self.collected_rewards.items():
            if name == "environment":
                self.cumulative_returns_since_policy_update.append(
                    rewards.get(agent_id, 0)
                )
                stats.stats_reporter.add_stat(
                    self.summary_path,
                    "Environment/Cumulative Reward",
                    rewards.get(agent_id, 0),
                )
                self.reward_buffer.appendleft(rewards.get(agent_id, 0))
                rewards[agent_id] = 0
            else:
                stats.stats_reporter.add_stat(
                    self.summary_path,
                    self.policy.reward_signals[name].stat_name,
                    rewards.get(agent_id, 0),
                )
                rewards[agent_id] = 0

    def clear_update_buffer(self) -> None:
        """
        Clear the buffers that have been built up during inference. If
        we're not training, this should be called instead of update_policy.
        """
        self.update_buffer.reset_agent()

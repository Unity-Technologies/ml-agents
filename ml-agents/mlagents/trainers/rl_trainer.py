# # Unity ML-Agents Toolkit
import logging
from typing import Dict
from collections import defaultdict

from mlagents.trainers.tf_policy import TFPolicy
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trainer import Trainer, UnityTrainerException
from mlagents.trainers.components.reward_signals import RewardSignalResult

LOGGER = logging.getLogger("mlagents.trainers")

RewardSignalResults = Dict[str, RewardSignalResult]


class RLTrainer(Trainer):  # pylint: disable=abstract-method
    """
    This class is the base class for trainers that use Reward Signals.
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
        self.collected_rewards: Dict[str, Dict[str, int]] = {
            "environment": defaultdict(lambda: 0)
        }
        self.update_buffer: AgentBuffer = AgentBuffer()
        self.episode_steps: Dict[str, int] = defaultdict(lambda: 0)

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

    def _update_end_episode_stats(self, agent_id: str, policy: TFPolicy) -> None:
        self.episode_steps[agent_id] = 0
        for name, rewards in self.collected_rewards.items():
            if name == "environment":
                self.cumulative_returns_since_policy_update.append(
                    rewards.get(agent_id, 0)
                )
                self.reward_buffer.appendleft(rewards.get(agent_id, 0))
                rewards[agent_id] = 0
            else:
                self.stats_reporter.add_stat(
                    policy.reward_signals[name].stat_name, rewards.get(agent_id, 0)
                )
                rewards[agent_id] = 0

    def clear_update_buffer(self) -> None:
        """
        Clear the buffers that have been built up during inference.
        """
        self.update_buffer.reset_agent()

    def advance(self) -> None:
        """
        Steps the trainer, taking in trajectories and updates if ready
        """
        super().advance()
        if not self.should_still_train:
            self.clear_update_buffer()

# # Unity ML-Agents Toolkit
import logging
from typing import Dict, NamedTuple
from collections import defaultdict
import numpy as np

from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trainer import Trainer, UnityTrainerException
from mlagents.trainers.components.reward_signals import RewardSignalResult

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
        self.episode_steps = {}

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

    def clear_update_buffer(self) -> None:
        """
        Clear the buffers that have been built up during inference. If
        we're not training, this should be called instead of update_policy.
        """
        self.update_buffer.reset_agent()

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        """
        Takes the output of the last action and store it into the training buffer.
        We break this out from add_experiences since it is very highly dependent
        on the type of trainer.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        :param agent_id: the Agent we're adding to.
        :param agent_idx: the index of the Agent agent_id
        """
        raise UnityTrainerException(
            "The add_policy_outputs method was not implemented."
        )

    def add_rewards_outputs(
        self,
        rewards_out: AllRewardsOutput,
        values: Dict[str, np.ndarray],
        agent_id: str,
        agent_idx: int,
        agent_next_idx: int,
    ) -> None:
        """
        Takes the value and evaluated rewards output of the last action and store it
        into the training buffer. We break this out from add_experiences since it is very
        highly dependent on the type of trainer.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        :param rewards_dict: Dict of rewards after evaluation
        :param agent_id: the Agent we're adding to.
        :param agent_idx: the index of the Agent agent_id in the current brain info
        :param agent_next_idx: the index of the Agent agent_id in the next brain info
        """
        raise UnityTrainerException(
            "The add_rewards_outputs method was not implemented."
        )

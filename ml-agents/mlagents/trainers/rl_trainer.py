# # Unity ML-Agents Toolkit
import logging
from typing import Dict, List, Any, NamedTuple
import numpy as np

from mlagents.envs.brain import AllBrainInfo, BrainInfo
from mlagents.envs.action_info import ActionInfoOutputs
from mlagents.trainers.buffer import Buffer
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
        self.collected_rewards = {"environment": {}}
        self.training_buffer = Buffer()
        self.episode_steps = {}

    def construct_curr_info(self, next_info: BrainInfo) -> BrainInfo:
        """
        Constructs a BrainInfo which contains the most recent previous experiences for all agents
        which correspond to the agents in a provided next_info.
        :BrainInfo next_info: A t+1 BrainInfo.
        :return: curr_info: Reconstructed BrainInfo to match agents of next_info.
        """
        visual_observations: List[List[Any]] = [
            [] for _ in next_info.visual_observations
        ]  # TODO add types to brain.py methods
        vector_observations = []
        text_observations = []
        memories = []
        rewards = []
        local_dones = []
        max_reacheds = []
        agents = []
        prev_vector_actions = []
        prev_text_actions = []
        action_masks = []
        for agent_id in next_info.agents:
            agent_brain_info = self.training_buffer[agent_id].last_brain_info
            if agent_brain_info is None:
                agent_brain_info = next_info
            agent_index = agent_brain_info.agents.index(agent_id)
            for i in range(len(next_info.visual_observations)):
                visual_observations[i].append(
                    agent_brain_info.visual_observations[i][agent_index]
                )
            vector_observations.append(
                agent_brain_info.vector_observations[agent_index]
            )
            text_observations.append(agent_brain_info.text_observations[agent_index])
            if self.policy.use_recurrent:
                if len(agent_brain_info.memories) > 0:
                    memories.append(agent_brain_info.memories[agent_index])
                else:
                    memories.append(self.policy.make_empty_memory(1))
            rewards.append(agent_brain_info.rewards[agent_index])
            local_dones.append(agent_brain_info.local_done[agent_index])
            max_reacheds.append(agent_brain_info.max_reached[agent_index])
            agents.append(agent_brain_info.agents[agent_index])
            prev_vector_actions.append(
                agent_brain_info.previous_vector_actions[agent_index]
            )
            prev_text_actions.append(
                agent_brain_info.previous_text_actions[agent_index]
            )
            action_masks.append(agent_brain_info.action_masks[agent_index])
        # Check if memories exists (i.e. next_info is not empty) before attempting vstack
        if self.policy.use_recurrent and memories:
            memories = np.vstack(memories)
        curr_info = BrainInfo(
            visual_observations,
            vector_observations,
            text_observations,
            memories,
            rewards,
            agents,
            local_dones,
            prev_vector_actions,
            prev_text_actions,
            max_reacheds,
            action_masks,
        )
        return curr_info

    def add_experiences(
        self,
        curr_all_info: AllBrainInfo,
        next_all_info: AllBrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param next_all_info: Dictionary of all current brains and corresponding BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        self.trainer_metrics.start_experience_collection_timer()
        if take_action_outputs:
            self.stats["Policy/Entropy"].append(take_action_outputs["entropy"].mean())
            self.stats["Policy/Learning Rate"].append(
                take_action_outputs["learning_rate"]
            )
            for name, signal in self.policy.reward_signals.items():
                self.stats[signal.value_name].append(
                    np.mean(take_action_outputs["value_heads"][name])
                )

        curr_info = curr_all_info[self.brain_name]
        next_info = next_all_info[self.brain_name]

        for agent_id in curr_info.agents:
            self.training_buffer[agent_id].last_brain_info = curr_info
            self.training_buffer[
                agent_id
            ].last_take_action_outputs = take_action_outputs

        if curr_info.agents != next_info.agents:
            curr_to_use = self.construct_curr_info(next_info)
        else:
            curr_to_use = curr_info

        # Evaluate and store the reward signals
        tmp_reward_signal_outs = {}
        for name, signal in self.policy.reward_signals.items():
            tmp_reward_signal_outs[name] = signal.evaluate(curr_to_use, next_info)
        # Store the environment reward
        tmp_environment = np.array(next_info.rewards)

        rewards_out = AllRewardsOutput(
            reward_signals=tmp_reward_signal_outs, environment=tmp_environment
        )

        for agent_id in next_info.agents:
            stored_info = self.training_buffer[agent_id].last_brain_info
            stored_take_action_outputs = self.training_buffer[
                agent_id
            ].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        self.training_buffer[agent_id]["visual_obs%d" % i].append(
                            stored_info.visual_observations[i][idx]
                        )
                        self.training_buffer[agent_id]["next_visual_obs%d" % i].append(
                            next_info.visual_observations[i][next_idx]
                        )
                    if self.policy.use_vec_obs:
                        self.training_buffer[agent_id]["vector_obs"].append(
                            stored_info.vector_observations[idx]
                        )
                        self.training_buffer[agent_id]["next_vector_in"].append(
                            next_info.vector_observations[next_idx]
                        )
                    if self.policy.use_recurrent:
                        if stored_info.memories.shape[1] == 0:
                            stored_info.memories = np.zeros(
                                (len(stored_info.agents), self.policy.m_size)
                            )
                        self.training_buffer[agent_id]["memory"].append(
                            stored_info.memories[idx]
                        )

                    self.training_buffer[agent_id]["masks"].append(1.0)
                    self.training_buffer[agent_id]["done"].append(
                        next_info.local_done[next_idx]
                    )
                    # Add the outputs of the last eval
                    self.add_policy_outputs(stored_take_action_outputs, agent_id, idx)
                    # Store action masks if neccessary
                    if not self.policy.use_continuous_act:
                        self.training_buffer[agent_id]["action_mask"].append(
                            stored_info.action_masks[idx], padding_value=1
                        )
                    self.training_buffer[agent_id]["prev_action"].append(
                        stored_info.previous_vector_actions[idx]
                    )

                    values = stored_take_action_outputs["value_heads"]

                    # Add the value outputs if needed
                    self.add_rewards_outputs(
                        rewards_out, values, agent_id, idx, next_idx
                    )

                    for name, rewards in self.collected_rewards.items():
                        if agent_id not in rewards:
                            rewards[agent_id] = 0
                        if name == "environment":
                            # Report the reward from the environment
                            rewards[agent_id] += rewards_out.environment[next_idx]
                        else:
                            # Report the reward signals
                            rewards[agent_id] += rewards_out.reward_signals[
                                name
                            ].scaled_reward[next_idx]
                if not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1
        self.trainer_metrics.end_experience_collection_timer()

    def end_episode(self) -> None:
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        self.training_buffer.reset_local_buffers()
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
        self.training_buffer.reset_update_buffer()

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
            "The process_experiences method was not implemented."
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
            "The process_experiences method was not implemented."
        )

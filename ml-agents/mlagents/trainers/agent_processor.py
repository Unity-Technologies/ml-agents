from typing import List, Dict, DefaultDict
from collections import defaultdict
import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents.envs.policy import Policy
from mlagents.envs.exception import UnityException
from mlagents.envs.brain import BrainInfo
from mlagents.envs.action_info import ActionInfoOutputs


class AgentProcessorException(UnityException):
    """
    Related to errors with the AgentProcessor.
    """

    pass


class AgentProcessor:
    """
    AgentProcessor contains a dictionary of AgentBuffer. The AgentBuffers are indexed by agent_id.
    Buffer also contains an update_buffer that corresponds to the buffer used when updating the model.
    """

    def __init__(self, policy: Policy):
        self.agent_buffers: DefaultDict[str, AgentBuffer] = defaultdict(AgentBuffer)
        self.stats: Dict[str, List] = defaultdict(list)
        # Note: this is needed until we switch to AgentExperiences as the data input type.
        # We still need some info from the policy (memories, previous actions)
        # that really should be gathered by the env-manager.
        self.policy = policy
        self.episode_steps: Dict[str, int] = {}

    def __str__(self):
        return "local_buffers :\n{0}".format(
            "\n".join(
                [
                    "\tagent {0} :{1}".format(k, str(self.agent_buffers[k]))
                    for k in self.agent_buffers.keys()
                ]
            )
        )

    def reset_local_buffers(self) -> None:
        """
        Resets all the local local_buffers
        """
        agent_ids = list(self.agent_buffers.keys())
        for k in agent_ids:
            self.agent_buffers[k].reset_agent()

    def append_to_update_buffer(
        self,
        update_buffer: AgentBuffer,
        agent_id: str,
        key_list: List[str] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> None:
        """
        Appends the buffer of an agent to the update buffer.
        :param agent_id: The id of the agent which data will be appended
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = self.agent_buffers[agent_id].keys()
        if not self.agent_buffers[agent_id].check_length(key_list):
            raise AgentProcessorException(
                "The length of the fields {0} for agent {1} where not of same length".format(
                    key_list, agent_id
                )
            )
        for field_key in key_list:
            update_buffer[field_key].extend(
                self.agent_buffers[agent_id][field_key].get_batch(
                    batch_size=batch_size, training_length=training_length
                )
            )

    def append_all_agent_batch_to_update_buffer(
        self,
        update_buffer: AgentBuffer,
        key_list: List[str] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> None:
        """
        Appends the buffer of all agents to the update buffer.
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        for agent_id in self.agent_buffers.keys():
            self.append_to_update_buffer(
                update_buffer, agent_id, key_list, batch_size, training_length
            )

    def add_experiences(
        self,
        curr_info: BrainInfo,
        next_info: BrainInfo,
        take_action_outputs: ActionInfoOutputs,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param curr_info: current BrainInfo.
        :param next_info: next BrainInfo.
        :param take_action_outputs: The outputs of the Policy's get_action method.
        """
        if take_action_outputs:
            self.stats["Policy/Entropy"].append(take_action_outputs["entropy"].mean())
            self.stats["Policy/Learning Rate"].append(
                take_action_outputs["learning_rate"]
            )
            for name, values in take_action_outputs["value_heads"].items():
                self.stats[name].append(np.mean(values))

        for agent_id in curr_info.agents:
            self.agent_buffers[agent_id].last_brain_info = curr_info
            self.agent_buffers[agent_id].last_take_action_outputs = take_action_outputs

        # Store the environment reward
        tmp_environment = np.array(next_info.rewards)

        for agent_id in next_info.agents:
            stored_info = self.agent_buffers[agent_id].last_brain_info
            stored_take_action_outputs = self.agent_buffers[
                agent_id
            ].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        self.agent_buffers[agent_id]["visual_obs%d" % i].append(
                            stored_info.visual_observations[i][idx]
                        )
                        self.agent_buffers[agent_id]["next_visual_obs%d" % i].append(
                            next_info.visual_observations[i][next_idx]
                        )
                    if self.policy.use_vec_obs:
                        self.agent_buffers[agent_id]["vector_obs"].append(
                            stored_info.vector_observations[idx]
                        )
                        self.agent_buffers[agent_id]["next_vector_in"].append(
                            next_info.vector_observations[next_idx]
                        )
                    if self.policy.use_recurrent:
                        self.agent_buffers[agent_id]["memory"].append(
                            self.policy.retrieve_memories([agent_id])[0, :]
                        )

                    self.agent_buffers[agent_id]["masks"].append(1.0)
                    self.agent_buffers[agent_id]["done"].append(
                        next_info.local_done[next_idx]
                    )
                    # Add the outputs of the last eval
                    self.add_policy_outputs(stored_take_action_outputs, agent_id, idx)

                    # Store action masks if necessary. Eventually these will be
                    # None for continuous actions
                    if stored_info.action_masks[idx] is not None:
                        self.agent_buffers[agent_id]["action_mask"].append(
                            stored_info.action_masks[idx], padding_value=1
                        )

                    # TODO: This should be done by the env_manager, and put it in
                    # the AgentExperience
                    self.agent_buffers[agent_id]["prev_action"].append(
                        self.policy.retrieve_previous_action([agent_id])[0, :]
                    )

                    values = stored_take_action_outputs["value_heads"]

                    # Add the value outputs if needed
                    self.agent_buffers[agent_id]["environment_rewards"].append(
                        tmp_environment
                    )

                    for name, value in values.items():
                        self.agent_buffers[agent_id][
                            "{}_value_estimates".format(name)
                        ].append(value[idx][0])

                if not next_info.local_done[next_idx]:
                    if agent_id not in self.episode_steps:
                        self.episode_steps[agent_id] = 0
                    self.episode_steps[agent_id] += 1
        self.policy.save_previous_action(
            curr_info.agents, take_action_outputs["action"]
        )

    def add_policy_outputs(
        self, take_action_outputs: ActionInfoOutputs, agent_id: str, agent_idx: int
    ) -> None:
        """
        Takes the output of the last action and store it into the training buffer.
        """
        actions = take_action_outputs["action"]
        if self.policy.use_continuous_act:
            actions_pre = take_action_outputs["pre_action"]
            self.agent_buffers[agent_id]["actions_pre"].append(actions_pre[agent_idx])
            epsilons = take_action_outputs["random_normal_epsilon"]
            self.agent_buffers[agent_id]["random_normal_epsilon"].append(
                epsilons[agent_idx]
            )
        a_dist = take_action_outputs["log_probs"]
        # value is a dictionary from name of reward to value estimate of the value head
        self.agent_buffers[agent_id]["actions"].append(actions[agent_idx])
        self.agent_buffers[agent_id]["action_probs"].append(a_dist[agent_idx])

    def process_experiences(self):
        pass

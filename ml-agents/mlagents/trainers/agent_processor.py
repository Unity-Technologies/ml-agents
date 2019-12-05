from typing import List, Dict, NamedTuple, Iterable
from collections import defaultdict
import numpy as np

from mlagents.trainers.buffer import AgentBuffer, BufferException
from mlagents.trainers.trainer import Trainer
from mlagents.envs.exception import UnityException
from mlagents.envs.brain import BrainInfo
from mlagents.envs.action_info import ActionInfoOutputs


class AgentExperience(NamedTuple):
    obs: List[np.ndarray]
    reward: float
    done: bool
    action: np.array
    action_probs: np.ndarray
    action_pre: np.ndarray  # TODO: Remove this
    action_mask: np.array
    prev_action: np.ndarray
    epsilon: float
    memory: np.array
    agent_id: str


class SplitObservations(NamedTuple):
    vector_observations: np.ndarray
    visual_observations: List[np.ndarray]


class Trajectory(NamedTuple):
    steps: Iterable[AgentExperience]
    next_step: AgentExperience  # The next step after the trajectory. Used for GAE when time_horizon is reached.


class AgentProcessorException(UnityException):
    """
    Related to errors with the AgentProcessor.
    """

    pass


def split_obs(obs: List[np.ndarray]) -> SplitObservations:
    vis_obs_indices = []
    vec_obs_indices = []
    for index, observation in enumerate(obs):
        if len(observation.shape) == 1:
            vec_obs_indices.append(index)
        if len(observation.shape) == 3:
            vis_obs_indices.append(index)
    vec_obs = np.concatenate([obs[i] for i in vec_obs_indices], axis=0)
    vis_obs = [obs[i] for i in vis_obs_indices]
    return SplitObservations(vector_observations=vec_obs, visual_observations=vis_obs)


class AgentProcessor:
    """
    AgentProcessor contains a dictionary of AgentBuffer. The AgentBuffers are indexed by agent_id.
    Buffer also contains an update_buffer that corresponds to the buffer used when updating the model.
    """

    def __init__(self, trainer: Trainer):
        self.processing_buffer = ProcessingBuffer()
        self.stats: Dict[str, List] = defaultdict(list)
        # Note: this is needed until we switch to AgentExperiences as the data input type.
        # We still need some info from the policy (memories, previous actions)
        # that really should be gathered by the env-manager.
        self.policy = trainer.policy
        self.episode_steps: Dict[str, int] = {}
        self.time_horizon: int = trainer.parameters["time_horizon"]
        self.trainer = trainer

    def __str__(self):
        return "local_buffers :\n{0}".format(
            "\n".join(
                [
                    "\tagent {0} :{1}".format(k, str(self.processing_buffer[k]))
                    for k in self.processing_buffer.keys()
                ]
            )
        )

    def reset_local_buffers(self) -> None:
        """
        Resets all the local local_buffers
        """
        agent_ids = list(self.processing_buffer.keys())
        for k in agent_ids:
            self.processing_buffer[k].reset_agent()

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
            self.processing_buffer[agent_id].last_brain_info = curr_info
            self.processing_buffer[
                agent_id
            ].last_take_action_outputs = take_action_outputs

        # Store the environment reward
        tmp_environment = np.array(next_info.rewards)

        for agent_id in next_info.agents:
            stored_info = self.processing_buffer[agent_id].last_brain_info
            stored_take_action_outputs = self.processing_buffer[
                agent_id
            ].last_take_action_outputs
            if stored_info is not None:
                idx = stored_info.agents.index(agent_id)
                next_idx = next_info.agents.index(agent_id)
                if not stored_info.local_done[idx]:
                    for i, _ in enumerate(stored_info.visual_observations):
                        self.processing_buffer[agent_id]["visual_obs%d" % i].append(
                            stored_info.visual_observations[i][idx]
                        )
                        self.processing_buffer[agent_id][
                            "next_visual_obs%d" % i
                        ].append(next_info.visual_observations[i][next_idx])
                    if self.policy.use_vec_obs:
                        self.processing_buffer[agent_id]["vector_obs"].append(
                            stored_info.vector_observations[idx]
                        )
                        self.processing_buffer[agent_id]["next_vector_in"].append(
                            next_info.vector_observations[next_idx]
                        )
                    if self.policy.use_recurrent:
                        self.processing_buffer[agent_id]["memory"].append(
                            self.policy.retrieve_memories([agent_id])[0, :]
                        )

                    self.processing_buffer[agent_id]["masks"].append(1.0)
                    self.processing_buffer[agent_id]["done"].append(
                        next_info.local_done[next_idx]
                    )
                    # Add the outputs of the last eval
                    self.add_policy_outputs(stored_take_action_outputs, agent_id, idx)

                    # Store action masks if necessary. Eventually these will be
                    # None for continuous actions
                    if stored_info.action_masks[idx] is not None:
                        self.processing_buffer[agent_id]["action_mask"].append(
                            stored_info.action_masks[idx], padding_value=1
                        )

                    # TODO: This should be done by the env_manager, and put it in
                    # the AgentExperience
                    self.processing_buffer[agent_id]["prev_action"].append(
                        self.policy.retrieve_previous_action([agent_id])[0, :]
                    )

                    values = stored_take_action_outputs["value_heads"]

                    # Add the value outputs if needed
                    self.processing_buffer[agent_id]["environment_rewards"].append(
                        tmp_environment[next_idx]
                    )

                    for name, value in values.items():
                        self.processing_buffer[agent_id][
                            "{}_value_estimates".format(name)
                        ].append(value[idx][0])

                agent_actions = self.processing_buffer[agent_id]["actions"]
                if (
                    next_info.local_done[next_idx]
                    or len(agent_actions) > self.time_horizon
                ) and len(agent_actions) > 0:
                    trajectory = self.processing_buffer.agent_to_trajectory(
                        agent_id, training_length=self.policy.sequence_length
                    )
                    self.trainer.process_trajectory(trajectory)
                elif not next_info.local_done[next_idx]:
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
            self.processing_buffer[agent_id]["actions_pre"].append(
                actions_pre[agent_idx]
            )
            epsilons = take_action_outputs["random_normal_epsilon"]
            self.processing_buffer[agent_id]["random_normal_epsilon"].append(
                epsilons[agent_idx]
            )
        a_dist = take_action_outputs["log_probs"]
        # value is a dictionary from name of reward to value estimate of the value head
        self.processing_buffer[agent_id]["actions"].append(actions[agent_idx])
        self.processing_buffer[agent_id]["action_probs"].append(a_dist[agent_idx])

    def process_experiences(self):
        pass


class ProcessingBuffer(dict):
    """
    ProcessingBuffer contains a dictionary of AgentBuffer. The AgentBuffers are indexed by agent_id.
    """

    def __str__(self):
        return "local_buffers :\n{0}".format(
            "\n".join(["\tagent {0} :{1}".format(k, str(self[k])) for k in self.keys()])
        )

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = AgentBuffer()
        return super().__getitem__(key)

    def reset_local_buffers(self) -> None:
        """
        Resets all the local AgentBuffers.
        """
        for buf in self.values():
            buf.reset_agent()

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
        :param update_buffer: A reference to an AgentBuffer to append the agent's buffer to
        :param agent_id: The id of the agent which data will be appended
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = self[agent_id].keys()
        if not self[agent_id].check_length(key_list):
            raise BufferException(
                "The length of the fields {0} for agent {1} were not of same length".format(
                    key_list, agent_id
                )
            )
        for field_key in key_list:
            update_buffer[field_key].extend(
                self[agent_id][field_key].get_batch(
                    batch_size=batch_size, training_length=training_length
                )
            )

    def agent_to_trajectory(
        self,
        agent_id: str,
        key_list: List[str] = None,
        batch_size: int = None,
        training_length: int = None,
    ) -> Trajectory:
        """
        Creates a Trajectory containing the AgentExperiences belonging to agent agent_id.
        :param agent_id: The id of the agent which data will be appended
        :param key_list: The fields that must be added. If None: all fields will be appended.
        :param batch_size: The number of elements that must be appended. If None: All of them will be.
        :param training_length: The length of the samples that must be appended. If None: only takes one element.
        """
        if key_list is None:
            key_list = self[agent_id].keys()
        if not self[agent_id].check_length(key_list):
            raise BufferException(
                "The length of the fields {0} for agent {1} were not of same length".format(
                    key_list, agent_id
                )
            )
        # trajectory = Trajectory()
        trajectory_list: List[AgentExperience] = []
        for _exp in range(self[agent_id].num_experiences):
            obs = []

            if "vector_obs" in key_list:
                obs.append(self[agent_id]["vector_obs"][_exp])
            memory = self[agent_id]["memory"][_exp] if "memory" in key_list else None
            # Assemble AgentExperience
            experience = AgentExperience(
                obs=obs,
                reward=self[agent_id]["environment_rewards"][_exp],
                done=self[agent_id]["done"][_exp],
                action=self[agent_id]["actions"][_exp],
                action_probs=self[agent_id]["action_probs"][_exp],
                action_pre=self[agent_id]["actions_pre"][_exp],
                action_mask=self[agent_id]["action_mask"][_exp],
                prev_action=self[agent_id]["prev_action"][_exp],
                agent_id=agent_id,
                memory=memory,
                epsilon=self[agent_id]["random_normal_epsilon"][_exp],
            )
            trajectory_list.append(experience)
        trajectory = Trajectory(steps=trajectory_list, next_step=experience)
        return trajectory

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
        for agent_id in self.keys():
            self.append_to_update_buffer(
                update_buffer, agent_id, key_list, batch_size, training_length
            )

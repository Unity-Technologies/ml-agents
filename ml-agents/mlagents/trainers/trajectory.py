from typing import List, NamedTuple
import itertools
import numpy as np

from mlagents.trainers.buffer import (
    AgentBuffer,
    AgentBufferField,
    ObservationKeyPrefix,
    AgentBufferKey,
    BufferKey,
)
from mlagents_envs.base_env import ActionTuple
from mlagents.trainers.torch.action_log_probs import LogProbsTuple


class GroupmateStatus(NamedTuple):
    """
    Stores data related to an agent's teammate.
    """

    obs: List[np.ndarray]
    reward: float
    action: ActionTuple
    done: bool


class AgentExperience(NamedTuple):
    obs: List[np.ndarray]
    reward: float
    done: bool
    action: ActionTuple
    action_probs: LogProbsTuple
    action_mask: np.ndarray
    prev_action: np.ndarray
    interrupted: bool
    memory: np.ndarray
    group_status: List[GroupmateStatus]
    group_reward: float


class ObsUtil:
    @staticmethod
    def get_name_at(index: int) -> AgentBufferKey:
        """
        returns the name of the observation given the index of the observation
        """
        return ObservationKeyPrefix.OBSERVATION, index

    @staticmethod
    def get_name_at_next(index: int) -> AgentBufferKey:
        """
        returns the name of the next observation given the index of the observation
        """
        return ObservationKeyPrefix.NEXT_OBSERVATION, index

    @staticmethod
    def from_buffer(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        """
        Creates the list of observations from an AgentBuffer
        """
        result: List[np.array] = []
        for i in range(num_obs):
            result.append(batch[ObsUtil.get_name_at(i)])
        return result

    @staticmethod
    def from_buffer_next(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        """
        Creates the list of next observations from an AgentBuffer
        """
        result = []
        for i in range(num_obs):
            result.append(batch[ObsUtil.get_name_at_next(i)])
        return result


class GroupObsUtil:
    @staticmethod
    def get_name_at(index: int) -> AgentBufferKey:
        """
        returns the name of the observation given the index of the observation
        """
        return ObservationKeyPrefix.GROUP_OBSERVATION, index

    @staticmethod
    def get_name_at_next(index: int) -> AgentBufferKey:
        """
        returns the name of the next team observation given the index of the observation
        """
        return ObservationKeyPrefix.NEXT_GROUP_OBSERVATION, index

    @staticmethod
    def _padded_time_to_batch(
        agent_buffer_field: AgentBufferField,
    ) -> List[np.ndarray]:
        """
        Convert an AgentBufferField of List of obs, where one of the dimension is time and the other is number (e.g.
        in the case of a variable number of critic observations) to a List of obs, where time is in the batch dimension
        of the obs, and the List is the variable number of agents. For cases where there are varying number of agents,
        pad the non-existent agents with NaN.
        """
        # Find the first observation. This should be USUALLY O(1)
        obs_shape = None
        for _group_obs in agent_buffer_field:
            if _group_obs:
                obs_shape = _group_obs[0].shape
                break
        # If there were no critic obs at all
        if obs_shape is None:
            return []

        new_list = list(
            map(
                lambda x: np.asanyarray(x),
                itertools.zip_longest(
                    *agent_buffer_field, fillvalue=np.full(obs_shape, np.nan)
                ),
            )
        )

        return new_list

    @staticmethod
    def _transpose_list_of_lists(
        list_list: List[List[np.ndarray]],
    ) -> List[List[np.ndarray]]:
        return list(map(list, zip(*list_list)))

    @staticmethod
    def from_buffer(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        """
        Creates the list of observations from an AgentBuffer
        """
        separated_obs: List[np.array] = []
        for i in range(num_obs):
            separated_obs.append(
                GroupObsUtil._padded_time_to_batch(batch[GroupObsUtil.get_name_at(i)])
            )
        # separated_obs contains a List(num_obs) of Lists(num_agents), we want to flip
        # that and get a List(num_agents) of Lists(num_obs)
        result = GroupObsUtil._transpose_list_of_lists(separated_obs)
        return result

    @staticmethod
    def from_buffer_next(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        """
        Creates the list of observations from an AgentBuffer
        """
        separated_obs: List[np.array] = []
        for i in range(num_obs):
            separated_obs.append(
                GroupObsUtil._padded_time_to_batch(
                    batch[GroupObsUtil.get_name_at_next(i)]
                )
            )
        # separated_obs contains a List(num_obs) of Lists(num_agents), we want to flip
        # that and get a List(num_agents) of Lists(num_obs)
        result = GroupObsUtil._transpose_list_of_lists(separated_obs)
        return result


class Trajectory(NamedTuple):
    steps: List[AgentExperience]
    next_obs: List[
        np.ndarray
    ]  # Observation following the trajectory, for bootstrapping
    next_group_obs: List[List[np.ndarray]]
    agent_id: str
    behavior_id: str

    def to_agentbuffer(self) -> AgentBuffer:
        """
        Converts a Trajectory to an AgentBuffer
        :param trajectory: A Trajectory
        :returns: AgentBuffer. Note that the length of the AgentBuffer will be one
        less than the trajectory, as the next observation need to be populated from the last
        step of the trajectory.
        """
        agent_buffer_trajectory = AgentBuffer()
        obs = self.steps[0].obs
        for step, exp in enumerate(self.steps):
            is_last_step = step == len(self.steps) - 1
            if not is_last_step:
                next_obs = self.steps[step + 1].obs
            else:
                next_obs = self.next_obs

            num_obs = len(obs)
            for i in range(num_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at(i)].append(obs[i])
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)].append(next_obs[i])

            # Take care of teammate obs and actions
            teammate_continuous_actions, teammate_discrete_actions, teammate_rewards = (
                [],
                [],
                [],
            )
            for group_status in exp.group_status:
                teammate_rewards.append(group_status.reward)
                teammate_continuous_actions.append(group_status.action.continuous)
                teammate_discrete_actions.append(group_status.action.discrete)

            # Team actions
            agent_buffer_trajectory[BufferKey.GROUP_CONTINUOUS_ACTION].append(
                teammate_continuous_actions
            )
            agent_buffer_trajectory[BufferKey.GROUP_DISCRETE_ACTION].append(
                teammate_discrete_actions
            )
            agent_buffer_trajectory[BufferKey.GROUPMATE_REWARDS].append(
                teammate_rewards
            )
            agent_buffer_trajectory[BufferKey.GROUP_REWARD].append(exp.group_reward)

            # Next actions
            teammate_cont_next_actions = []
            teammate_disc_next_actions = []
            if not is_last_step:
                next_exp = self.steps[step + 1]
                for group_status in next_exp.group_status:
                    teammate_cont_next_actions.append(group_status.action.continuous)
                    teammate_disc_next_actions.append(group_status.action.discrete)
            else:
                for group_status in exp.group_status:
                    teammate_cont_next_actions.append(group_status.action.continuous)
                    teammate_disc_next_actions.append(group_status.action.discrete)

            agent_buffer_trajectory[BufferKey.GROUP_NEXT_CONT_ACTION].append(
                teammate_cont_next_actions
            )
            agent_buffer_trajectory[BufferKey.GROUP_NEXT_DISC_ACTION].append(
                teammate_disc_next_actions
            )

            for i in range(num_obs):
                ith_group_obs = []
                for _group_status in exp.group_status:
                    # Assume teammates have same obs space
                    ith_group_obs.append(_group_status.obs[i])
                agent_buffer_trajectory[GroupObsUtil.get_name_at(i)].append(
                    ith_group_obs
                )

                ith_group_obs_next = []
                if is_last_step:
                    for _obs in self.next_group_obs:
                        ith_group_obs_next.append(_obs[i])
                else:
                    next_group_status = self.steps[step + 1].group_status
                    for _group_status in next_group_status:
                        # Assume teammates have same obs space
                        ith_group_obs_next.append(_group_status.obs[i])
                agent_buffer_trajectory[GroupObsUtil.get_name_at_next(i)].append(
                    ith_group_obs_next
                )

            if exp.memory is not None:
                agent_buffer_trajectory[BufferKey.MEMORY].append(exp.memory)

            agent_buffer_trajectory[BufferKey.MASKS].append(1.0)
            agent_buffer_trajectory[BufferKey.DONE].append(exp.done)
            agent_buffer_trajectory[BufferKey.GROUP_DONES].append(
                [_status.done for _status in exp.group_status]
            )

            # Adds the log prob and action of continuous/discrete separately
            agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION].append(
                exp.action.continuous
            )
            agent_buffer_trajectory[BufferKey.DISCRETE_ACTION].append(
                exp.action.discrete
            )

            cont_next_actions = np.zeros_like(exp.action.continuous)
            disc_next_actions = np.zeros_like(exp.action.discrete)

            if not is_last_step:
                next_action = self.steps[step + 1].action
                cont_next_actions = next_action.continuous
                disc_next_actions = next_action.discrete

            agent_buffer_trajectory[BufferKey.NEXT_CONT_ACTION].append(
                cont_next_actions
            )
            agent_buffer_trajectory[BufferKey.NEXT_DISC_ACTION].append(
                disc_next_actions
            )

            agent_buffer_trajectory[BufferKey.CONTINUOUS_LOG_PROBS].append(
                exp.action_probs.continuous
            )
            agent_buffer_trajectory[BufferKey.DISCRETE_LOG_PROBS].append(
                exp.action_probs.discrete
            )

            # Store action masks if necessary. Note that 1 means active, while
            # in AgentExperience False means active.
            if exp.action_mask is not None:
                mask = 1 - np.concatenate(exp.action_mask)
                agent_buffer_trajectory[BufferKey.ACTION_MASK].append(
                    mask, padding_value=1
                )
            else:
                # This should never be needed unless the environment somehow doesn't supply the
                # action mask in a discrete space.

                action_shape = exp.action.discrete.shape
                agent_buffer_trajectory[BufferKey.ACTION_MASK].append(
                    np.ones(action_shape, dtype=np.float32), padding_value=1
                )
            agent_buffer_trajectory[BufferKey.PREV_ACTION].append(exp.prev_action)
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS].append(exp.reward)

            # Store the next visual obs as the current
            obs = next_obs
        return agent_buffer_trajectory

    @property
    def done_reached(self) -> bool:
        """
        Returns true if trajectory is terminated with a Done.
        """
        return self.steps[-1].done

    @property
    def teammate_dones_reached(self) -> bool:
        """
        Returns true if all teammates are done at the end of the trajectory.
        Combine with done_reached to check if the whole team is done.
        """
        return all(_status.done for _status in self.steps[-1].group_status)

    @property
    def interrupted(self) -> bool:
        """
        Returns true if trajectory was terminated because max steps was reached.
        """
        return self.steps[-1].interrupted

from typing import List, NamedTuple
import itertools
import attr
import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.base_env import ActionTuple
from mlagents.trainers.torch.action_log_probs import LogProbsTuple


@attr.s(auto_attribs=True)
class TeammateStatus:
    """
    Stores data related to an agent's teammate.
    """

    obs: List[np.ndarray]
    reward: float
    action: ActionTuple


@attr.s(auto_attribs=True)
class AgentExperience:
    obs: List[np.ndarray]
    teammate_status: List[TeammateStatus]
    reward: float
    done: bool
    action: ActionTuple
    action_probs: LogProbsTuple
    action_mask: np.ndarray
    prev_action: np.ndarray
    interrupted: bool
    memory: np.ndarray


class ObsUtil:
    @staticmethod
    def get_name_at(index: int) -> str:
        """
        returns the name of the observation given the index of the observation
        """
        return f"obs_{index}"

    @staticmethod
    def get_name_at_next(index: int) -> str:
        """
        returns the name of the next observation given the index of the observation
        """
        return f"next_obs_{index}"

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


class TeamObsUtil:
    @staticmethod
    def get_name_at(index: int) -> str:
        """
        returns the name of the observation given the index of the observation
        """
        return f"team_obs_{index}"

    @staticmethod
    def _padded_time_to_batch(
        agent_buffer_field: AgentBuffer.AgentBufferField,
    ) -> List[np.ndarray]:
        """
        Convert an AgentBufferField of List of obs, where one of the dimension is time and the other is number (e.g.
        in the case of a variable number of critic observations) to a List of obs, where time is in the batch dimension
        of the obs, and the List is the variable number of agents. For cases where there are varying number of agents,
        pad the non-existent agents with NaN.
        """
        # Find the first observation. This should be USUALLY O(1)
        obs_shape = None
        for _team_obs in agent_buffer_field:
            if _team_obs:
                obs_shape = _team_obs[0].shape
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
                TeamObsUtil._padded_time_to_batch(batch[TeamObsUtil.get_name_at(i)])
            )
        # separated_obs contains a List(num_obs) of Lists(num_agents), we want to flip
        # that and get a List(num_agents) of Lists(num_obs)
        result = TeamObsUtil._transpose_list_of_lists(separated_obs)
        return result


class Trajectory(NamedTuple):
    steps: List[AgentExperience]
    next_obs: List[
        np.ndarray
    ]  # Observation following the trajectory, for bootstrapping
    next_collab_obs: List[List[np.ndarray]]
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
            if step < len(self.steps) - 1:
                next_obs = self.steps[step + 1].obs
            else:
                next_obs = self.next_obs

            num_obs = len(obs)
            for i in range(num_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at(i)].append(obs[i])
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)].append(next_obs[i])

            teammate_continuous_actions, teammate_discrete_actions, teammate_rewards = (
                [],
                [],
                [],
            )
            for teammate_status in exp.teammate_status:
                teammate_rewards.append(teammate_status.reward)
                teammate_continuous_actions.append(teammate_status.action.continuous)
                teammate_discrete_actions.append(teammate_status.action.discrete)

            for i in range(num_obs):
                ith_team_obs = []
                for _teammate_status in exp.teammate_status:
                    # Assume teammates have same obs space
                    ith_team_obs.append(_teammate_status.obs[i])
                agent_buffer_trajectory[TeamObsUtil.get_name_at(i)].append(ith_team_obs)

            agent_buffer_trajectory["team_rewards"].append(teammate_rewards)

            if exp.memory is not None:
                agent_buffer_trajectory["memory"].append(exp.memory)

            agent_buffer_trajectory["masks"].append(1.0)
            agent_buffer_trajectory["done"].append(exp.done)

            # Adds the log prob and action of continuous/discrete separately
            agent_buffer_trajectory["continuous_action"].append(exp.action.continuous)
            agent_buffer_trajectory["discrete_action"].append(exp.action.discrete)

            # Team actions
            agent_buffer_trajectory["team_continuous_action"].append(
                teammate_continuous_actions
            )
            agent_buffer_trajectory["team_discrete_action"].append(
                teammate_discrete_actions
            )

            agent_buffer_trajectory["continuous_log_probs"].append(
                exp.action_probs.continuous
            )
            agent_buffer_trajectory["discrete_log_probs"].append(
                exp.action_probs.discrete
            )

            # Store action masks if necessary. Note that 1 means active, while
            # in AgentExperience False means active.
            if exp.action_mask is not None:
                mask = 1 - np.concatenate(exp.action_mask)
                agent_buffer_trajectory["action_mask"].append(mask, padding_value=1)
            else:
                # This should never be needed unless the environment somehow doesn't supply the
                # action mask in a discrete space.

                action_shape = exp.action.discrete.shape
                agent_buffer_trajectory["action_mask"].append(
                    np.ones(action_shape, dtype=np.float32), padding_value=1
                )
            agent_buffer_trajectory["prev_action"].append(exp.prev_action)
            agent_buffer_trajectory["environment_rewards"].append(exp.reward)

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
    def interrupted(self) -> bool:
        """
        Returns true if trajectory was terminated because max steps was reached.
        """
        return self.steps[-1].interrupted

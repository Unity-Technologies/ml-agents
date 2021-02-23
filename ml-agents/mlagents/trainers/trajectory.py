from typing import List, NamedTuple
import numpy as np

from mlagents.trainers.buffer import (
    AgentBuffer,
    ObservationKeyPrefix,
    AgentBufferKey,
    BufferKey,
)
from mlagents_envs.base_env import ActionTuple
from mlagents.trainers.torch.action_log_probs import LogProbsTuple


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


class Trajectory(NamedTuple):
    steps: List[AgentExperience]
    next_obs: List[
        np.ndarray
    ]  # Observation following the trajectory, for bootstrapping
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

            if exp.memory is not None:
                agent_buffer_trajectory[BufferKey.MEMORY].append(exp.memory)

            agent_buffer_trajectory[BufferKey.MASKS].append(1.0)
            agent_buffer_trajectory[BufferKey.DONE].append(exp.done)

            # Adds the log prob and action of continuous/discrete separately
            agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION].append(
                exp.action.continuous
            )
            agent_buffer_trajectory[BufferKey.DISCRETE_ACTION].append(
                exp.action.discrete
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
    def interrupted(self) -> bool:
        """
        Returns true if trajectory was terminated because max steps was reached.
        """
        return self.steps[-1].interrupted

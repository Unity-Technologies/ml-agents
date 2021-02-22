import abc

import numpy as np

from typing import List, NamedTuple

from mlagents_envs.base_env import ActionTuple, BehaviorSpec

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.trajectory import ObsUtil


class DemonstrationExperience(NamedTuple):
    obs: List[np.ndarray]
    reward: float
    done: bool
    action: ActionTuple
    prev_action: np.ndarray
    interrupted: bool


class DemonstrationTrajectory(NamedTuple):
    experiences: List[DemonstrationExperience]

    def to_agentbuffer(self) -> AgentBuffer:
        """
        Converts a Trajectory to an AgentBuffer
        :param trajectory: A Trajectory
        :returns: AgentBuffer. Note that the length of the AgentBuffer will be one
        less than the trajectory, as the next observation need to be populated from the last
        step of the trajectory.
        """
        agent_buffer_trajectory = AgentBuffer()
        for exp in self.experiences:
            for i, obs in enumerate(exp.obs):
                agent_buffer_trajectory[ObsUtil.get_name_at(i)].append(obs)

            # TODO Not in demo_loader
            agent_buffer_trajectory[BufferKey.MASKS].append(1.0)
            agent_buffer_trajectory[BufferKey.DONE].append(exp.done)

            agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION].append(
                exp.action.continuous
            )
            agent_buffer_trajectory[BufferKey.DISCRETE_ACTION].append(
                exp.action.discrete
            )

            agent_buffer_trajectory[BufferKey.PREV_ACTION].append(exp.prev_action)
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS].append(exp.reward)

        return agent_buffer_trajectory


class DemonstrationProvider(abc.ABC):
    @abc.abstractmethod
    def get_behavior_spec(self) -> BehaviorSpec:
        pass

    @abc.abstractmethod
    def pop_trajectories(self) -> List[DemonstrationTrajectory]:
        pass

    def to_agentbuffer(self, training_length: int) -> AgentBuffer:
        buffer_out = AgentBuffer()
        trajectories = self.pop_trajectories()
        for trajectory in trajectories:
            temp_buffer = trajectory.to_agentbuffer()
            temp_buffer.resequence_and_append(
                buffer_out, batch_size=None, training_length=training_length
            )
        return buffer_out

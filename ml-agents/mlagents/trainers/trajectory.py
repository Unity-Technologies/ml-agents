from typing import List, NamedTuple
import numpy as np

from mlagents.trainers.buffer import AgentBuffer


class AgentExperience(NamedTuple):
    obs: List[np.ndarray]
    reward: float
    done: bool
    action: np.ndarray
    action_probs: np.ndarray
    action_pre: np.ndarray  # TODO: Remove this
    action_mask: np.ndarray
    prev_action: np.ndarray
    interrupted: bool
    memory: np.ndarray


class ObsUtil:
    @staticmethod
    def get_obs_with_rank(observations: List[np.array], rank: int) -> List[np.array]:
        result: List[np.array] = []
        for obs in observations:
            if len(obs.shape) == rank:
                result += [obs]
        return result

    @staticmethod
    def get_name_at(index: int) -> str:
        return "obs_%d" % index

    @staticmethod
    def get_name_at_next(index: int) -> str:
        return "next_obs_%d" % index

    @staticmethod
    def from_buffer(batch: AgentBuffer, num_obs: int) -> List[np.array]:
        result: List[np.array] = []
        for i in range(num_obs):
            result.append(batch[ObsUtil.get_name_at(i)])
        return result

    @staticmethod
    def from_buffer_next(batch: AgentBuffer, num_obs: int) -> List[np.array]:
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
                agent_buffer_trajectory["memory"].append(exp.memory)

            agent_buffer_trajectory["masks"].append(1.0)
            agent_buffer_trajectory["done"].append(exp.done)
            # Add the outputs of the last eval
            if exp.action_pre is not None:
                actions_pre = exp.action_pre
                agent_buffer_trajectory["actions_pre"].append(actions_pre)

            # value is a dictionary from name of reward to value estimate of the value head
            agent_buffer_trajectory["actions"].append(exp.action)
            agent_buffer_trajectory["action_probs"].append(exp.action_probs)

            # Store action masks if necessary. Note that 1 means active, while
            # in AgentExperience False means active.
            if exp.action_mask is not None:
                mask = 1 - np.concatenate(exp.action_mask)
                agent_buffer_trajectory["action_mask"].append(mask, padding_value=1)
            else:
                # This should never be needed unless the environment somehow doesn't supply the
                # action mask in a discrete space.
                agent_buffer_trajectory["action_mask"].append(
                    np.ones(exp.action_probs.shape, dtype=np.float32), padding_value=1
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

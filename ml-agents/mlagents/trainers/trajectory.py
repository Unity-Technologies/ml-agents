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
    max_step: bool
    memory: np.ndarray


class SplitObservations(NamedTuple):
    vector_observations: np.ndarray
    visual_observations: List[np.ndarray]

    @staticmethod
    def from_observations(obs: List[np.ndarray]) -> "SplitObservations":
        """
        Divides a List of numpy arrays into a SplitObservations NamedTuple.
        This allows you to access the vector and visual observations directly,
        without enumerating the list over and over.
        :param obs: List of numpy arrays (observation)
        :returns: A SplitObservations object.
        """
        vis_obs_list: List[np.ndarray] = []
        vec_obs_list: List[np.ndarray] = []
        for observation in obs:
            if len(observation.shape) == 1:
                vec_obs_list.append(observation)
            if len(observation.shape) == 3:
                vis_obs_list.append(observation)
        vec_obs = (
            np.concatenate(vec_obs_list, axis=0)
            if len(vec_obs_list) > 0
            else np.array([], dtype=np.float32)
        )
        return SplitObservations(
            vector_observations=vec_obs, visual_observations=vis_obs_list
        )


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
        vec_vis_obs = SplitObservations.from_observations(self.steps[0].obs)
        for step, exp in enumerate(self.steps):
            if step < len(self.steps) - 1:
                next_vec_vis_obs = SplitObservations.from_observations(
                    self.steps[step + 1].obs
                )
            else:
                next_vec_vis_obs = SplitObservations.from_observations(self.next_obs)

            for i, _ in enumerate(vec_vis_obs.visual_observations):
                agent_buffer_trajectory["visual_obs%d" % i].append(
                    vec_vis_obs.visual_observations[i]
                )
                agent_buffer_trajectory["next_visual_obs%d" % i].append(
                    next_vec_vis_obs.visual_observations[i]
                )
            agent_buffer_trajectory["vector_obs"].append(
                vec_vis_obs.vector_observations
            )
            agent_buffer_trajectory["next_vector_in"].append(
                next_vec_vis_obs.vector_observations
            )
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

            # Store action masks if necessary. Eventually these will be
            # None for continuous actions
            if exp.action_mask is not None:
                agent_buffer_trajectory["action_mask"].append(
                    exp.action_mask, padding_value=1
                )

            agent_buffer_trajectory["prev_action"].append(exp.prev_action)
            agent_buffer_trajectory["environment_rewards"].append(exp.reward)

            # Store the next visual obs as the current
            vec_vis_obs = next_vec_vis_obs
        return agent_buffer_trajectory

    @property
    def done_reached(self) -> bool:
        """
        Returns true if trajectory is terminated with a Done.
        """
        return self.steps[-1].done

    @property
    def max_step_reached(self) -> bool:
        """
        Returns true if trajectory was terminated because max steps was reached.
        """
        return self.steps[-1].max_step

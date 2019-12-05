from typing import List, NamedTuple
import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents.envs.exception import UnityException


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


class BootstrapExperience(NamedTuple):
    """
    A partial AgentExperience needed to bootstrap GAE.
    """

    obs: List[np.ndarray]
    agent_id: str


class SplitObservations(NamedTuple):
    vector_observations: np.ndarray
    visual_observations: List[np.ndarray]


class Trajectory(NamedTuple):
    steps: List[AgentExperience]
    bootstrap_step: BootstrapExperience  # The next step after the trajectory. Used for GAE.


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


def trajectory_to_agentbuffer(trajectory: Trajectory) -> AgentBuffer:
    """
    Converts a Trajectory to an AgentBuffer
    :param trajectory: A Trajectory
    :returns: AgentBuffer
    """
    agent_buffer_trajectory = AgentBuffer()
    for step, exp in enumerate(trajectory.steps):
        vec_vis_obs = split_obs(exp.obs)
        if step < len(trajectory.steps) - 1:
            next_vec_vis_obs = split_obs(trajectory.steps[step + 1].obs)
        else:
            next_vec_vis_obs = split_obs(trajectory.bootstrap_step.obs)
        for i, _ in enumerate(vec_vis_obs.visual_observations):
            agent_buffer_trajectory["visual_obs%d" % i].append(
                vec_vis_obs.visual_observations[i]
            )
            agent_buffer_trajectory["next_visual_obs%d" % i].append(
                next_vec_vis_obs.visual_observations[i]
            )
        if vec_vis_obs.vector_observations.size > 0:
            agent_buffer_trajectory["vector_obs"].append(
                vec_vis_obs.vector_observations
            )
            agent_buffer_trajectory["next_vector_in"].append(
                next_vec_vis_obs.vector_observations
            )
        if exp.memory:
            agent_buffer_trajectory["memory"].append(exp.memory)

        agent_buffer_trajectory["masks"].append(1.0)
        agent_buffer_trajectory["done"].append(exp.done)
        # Add the outputs of the last eval
        if exp.action_pre is not None:
            actions_pre = exp.action_pre
            agent_buffer_trajectory["actions_pre"].append(actions_pre)
        if exp.epsilon is not None:
            epsilons = exp.epsilon
            agent_buffer_trajectory["random_normal_epsilon"].append(epsilons)
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

        # Add the value outputs if needed
        agent_buffer_trajectory["environment_rewards"].append(exp.reward)
    return agent_buffer_trajectory

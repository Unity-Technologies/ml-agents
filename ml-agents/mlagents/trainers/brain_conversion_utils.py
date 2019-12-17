from mlagents.trainers.brain import BrainInfo, BrainParameters, CameraResolution
from mlagents_envs.base_env import BatchedStepResult, AgentGroupSpec
from mlagents_envs.exception import UnityEnvironmentException
import numpy as np
from typing import List


def step_result_to_brain_info(
    step_result: BatchedStepResult,
    group_spec: AgentGroupSpec,
    agent_id_prefix: int = None,
) -> BrainInfo:
    n_agents = step_result.n_agents()
    vis_obs_indices = []
    vec_obs_indices = []
    for index, observation in enumerate(step_result.obs):
        if len(observation.shape) == 2:
            vec_obs_indices.append(index)
        elif len(observation.shape) == 4:
            vis_obs_indices.append(index)
        else:
            raise UnityEnvironmentException(
                "Invalid input received from the environment, the observation should "
                "either be a vector of float or a PNG image"
            )
    if len(vec_obs_indices) == 0:
        vec_obs = np.zeros((n_agents, 0), dtype=np.float32)
    else:
        vec_obs = np.concatenate([step_result.obs[i] for i in vec_obs_indices], axis=1)
    vis_obs = [step_result.obs[i] for i in vis_obs_indices]
    mask = np.ones((n_agents, np.sum(group_spec.action_size)), dtype=np.float32)
    if group_spec.is_action_discrete():
        mask = np.ones(
            (n_agents, np.sum(group_spec.discrete_action_branches)), dtype=np.float32
        )
        if step_result.action_mask is not None:
            mask = 1 - np.concatenate(step_result.action_mask, axis=1)
    if agent_id_prefix is None:
        agent_ids = [str(ag_id) for ag_id in list(step_result.agent_id)]
    else:
        agent_ids = [f"${agent_id_prefix}-{ag_id}" for ag_id in step_result.agent_id]
    return BrainInfo(
        vis_obs,
        vec_obs,
        list(step_result.reward),
        agent_ids,
        list(step_result.done),
        list(step_result.max_step),
        mask,
    )


def group_spec_to_brain_parameters(
    name: str, group_spec: AgentGroupSpec
) -> BrainParameters:
    vec_size = np.sum(
        [shape[0] for shape in group_spec.observation_shapes if len(shape) == 1]
    )
    vis_sizes = [shape for shape in group_spec.observation_shapes if len(shape) == 3]
    cam_res = [CameraResolution(s[0], s[1], s[2]) for s in vis_sizes]
    a_size: List[int] = []
    if group_spec.is_action_discrete():
        a_size += list(group_spec.discrete_action_branches)
        vector_action_space_type = 0
    else:
        a_size += [group_spec.action_size]
        vector_action_space_type = 1
    return BrainParameters(
        name, int(vec_size), cam_res, a_size, [], vector_action_space_type
    )

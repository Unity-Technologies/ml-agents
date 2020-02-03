from mlagents.trainers.brain import BrainParameters, CameraResolution
from mlagents_envs.base_env import AgentGroupSpec
import numpy as np
from typing import List


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


def get_global_agent_id(worker_id: int, agent_id: int) -> str:
    """
    Create an agent id that is unique across environment workers using the worker_id.
    """
    return f"${worker_id}-{agent_id}"

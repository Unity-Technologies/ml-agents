import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.trajectory import SplitObservations


def create_agent_buffer(
    behavior_spec: BehaviorSpec, number: int, reward: float = 0.0
) -> AgentBuffer:
    buffer = AgentBuffer()
    curr_observations = [
        np.random.normal(size=shape).astype(np.float32)
        for shape in behavior_spec.observation_shapes
    ]
    next_observations = [
        np.random.normal(size=shape).astype(np.float32)
        for shape in behavior_spec.observation_shapes
    ]
    action_buffer = behavior_spec.action_spec.random_action(1)
    action = {}
    if behavior_spec.action_spec.continuous_size > 0:
        action["continuous_action"] = action_buffer.continuous
    if behavior_spec.action_spec.discrete_size > 0:
        action["discrete_action"] = action_buffer.discrete

    for _ in range(number):
        curr_split_obs = SplitObservations.from_observations(curr_observations)
        next_split_obs = SplitObservations.from_observations(next_observations)
        for i, _ in enumerate(curr_split_obs.visual_observations):
            buffer["visual_obs%d" % i].append(curr_split_obs.visual_observations[i])
            buffer["next_visual_obs%d" % i].append(
                next_split_obs.visual_observations[i]
            )
        buffer["vector_obs"].append(curr_split_obs.vector_observations)
        buffer["next_vector_in"].append(next_split_obs.vector_observations)
        for _act_type, _act in action.items():
            buffer[_act_type].append(_act[0, :])
        buffer["reward"].append(np.ones(1, dtype=np.float32) * reward)
        buffer["masks"].append(np.ones(1, dtype=np.float32))
    buffer["done"] = np.zeros(number, dtype=np.float32)
    return buffer

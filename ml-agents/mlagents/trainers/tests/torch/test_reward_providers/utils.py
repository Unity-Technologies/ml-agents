import numpy as np
from mlagents.trainers.buffer import AgentBuffer
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.trajectory import ObsUtil


def create_agent_buffer(
    behavior_spec: BehaviorSpec, number: int, reward: float = 0.0
) -> AgentBuffer:
    buffer = AgentBuffer()
    curr_obs = [
        np.random.normal(size=shape).astype(np.float32)
        for shape in behavior_spec.observation_shapes
    ]
    next_obs = [
        np.random.normal(size=shape).astype(np.float32)
        for shape in behavior_spec.observation_shapes
    ]
    action = behavior_spec.action_spec.random_action(1)[0, :]
    for _ in range(number):
        for i, obs in enumerate(curr_obs):
            buffer[ObsUtil.get_name_at(i)].append(obs)
        for i, obs in enumerate(next_obs):
            buffer[ObsUtil.get_name_at_next(i)].append(obs)
        buffer["actions"].append(action)
        buffer["reward"].append(np.ones(1, dtype=np.float32) * reward)
        buffer["masks"].append(np.ones(1, dtype=np.float32))
    buffer["done"] = np.zeros(number, dtype=np.float32)
    return buffer

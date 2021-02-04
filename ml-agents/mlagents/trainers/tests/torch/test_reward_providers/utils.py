import numpy as np
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.trajectory import ObsUtil


def create_agent_buffer(
    behavior_spec: BehaviorSpec, number: int, reward: float = 0.0
) -> AgentBuffer:
    buffer = AgentBuffer()
    curr_obs = [
        np.random.normal(size=obs_spec.shape).astype(np.float32)
        for obs_spec in behavior_spec.observation_specs
    ]
    next_obs = [
        np.random.normal(size=obs_spec.shape).astype(np.float32)
        for obs_spec in behavior_spec.observation_specs
    ]
    action_buffer = behavior_spec.action_spec.random_action(1)
    action = {}
    if behavior_spec.action_spec.continuous_size > 0:
        action[BufferKey.CONTINUOUS_ACTION] = action_buffer.continuous
    if behavior_spec.action_spec.discrete_size > 0:
        action[BufferKey.DISCRETE_ACTION] = action_buffer.discrete

    for _ in range(number):
        for i, obs in enumerate(curr_obs):
            buffer[ObsUtil.get_name_at(i)].append(obs)
        for i, obs in enumerate(next_obs):
            buffer[ObsUtil.get_name_at_next(i)].append(obs)
        # TODO
        # buffer[AgentBufferKey.ACTIONS].append(action)
        for _act_type, _act in action.items():
            buffer[_act_type].append(_act[0, :])
        # TODO was "rewards"
        buffer[BufferKey.ENVIRONMENT_REWARDS].append(
            np.ones(1, dtype=np.float32) * reward
        )
        buffer[BufferKey.MASKS].append(np.ones(1, dtype=np.float32))
    buffer[BufferKey.DONE] = np.zeros(number, dtype=np.float32)
    return buffer

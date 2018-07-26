from gym.envs.registration import register

register(
    id='unityenv-v0',
    entry_point='gym_unity.envs:UnityEnv',
)
register(
    id='unityenv-multiagent-v0',
    entry_point='gym_unity.envs:UnityMultiAgentEnv',
)

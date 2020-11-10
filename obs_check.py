from mlagents_envs.environment import UnityEnvironment


# Getting observation types
env = UnityEnvironment()
env.reset()
bspec = list(env.behavior_specs.values())[0]
print(bspec.observation_shapes)

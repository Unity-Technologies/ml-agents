from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.agent_parameters_channel import AgentParametersChannel


# Getting observation types
agent_params = AgentParametersChannel()
env = UnityEnvironment(side_channels=[agent_params])
env.reset()
bspec = list(env.behavior_specs.values())[0]
print(bspec.sensor_types)
dsteps, tsteps = env.get_steps(list(env.behavior_specs.keys())[0])
print(dsteps.obs)

# Sending agent parameterizations
for i, _id in enumerate(dsteps.agent_id):
    agent_params.set_float_parameter(_id, "test_param", i * 1000)
env.reset()
env.step()

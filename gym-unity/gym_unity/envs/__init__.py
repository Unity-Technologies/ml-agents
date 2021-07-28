from gym.envs.registration import register
from gym.error import Error
from mlagents_envs.registry import default_registry
from gym_unity.envs.gym_env import UnityToGymWrapper


from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.exception import UnityWorkerInUseException


def gym_entry_point(env_name):
    def create(action_space_seed=None, **kwargs):
        if "side_channels" not in kwargs:
            kwargs["side_channels"] = [
                EngineConfigurationChannel(),
                EnvironmentParametersChannel(),
                StatsSideChannel(),
            ]
        _e = None
        if "base_port" not in kwargs:
            port = 6000
            while _e is None:
                try:
                    kwargs["base_port"] = port
                    _e = default_registry[env_name].make(**kwargs)
                except UnityWorkerInUseException:
                    port += 1
                    pass
        else:
            _e = default_registry[env_name].make(**kwargs)
        return UnityToGymWrapper(_e, action_space_seed)

    return create


for key in default_registry:
    registry_key = key + "-v0"
    try:
        register(id=registry_key, entry_point=gym_entry_point(key))
    except Error:
        pass

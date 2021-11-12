from mlagents_envs.registry import default_registry

from typing import Optional
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs import logging_util

from pettingzoo_unity.envs.unity_base_env import UnityBaseEnv  # noqa
from pettingzoo_unity.envs.unity_aec_env import UnityAECEnv
from pettingzoo_unity.envs.unity_parallel_env import UnityParallelEnv  # noqa

logger = logging_util.get_logger(__name__)


class PettingZooEnv:
    def __init__(self, env_id: str) -> None:
        self.env_id = env_id

    def env(self, seed: Optional[int] = None, **kwargs) -> UnityAECEnv:
        """
        Creates the environment with env_id from unity's default_registry and wraps it in a UnityToPettingZooWrapper
        :param seed: The seed for the action spaces of the agents.
        :param kwargs: Any argument accepted by `UnityEnvironment`class except file_name
        """
        # If not side_channels specified, add the followings
        if "side_channels" not in kwargs:
            kwargs["side_channels"] = [
                EngineConfigurationChannel(),
                EnvironmentParametersChannel(),
                StatsSideChannel(),
            ]
        _env = None
        # If no base port argument is provided, try ports starting at 6000 until one is free
        if "base_port" not in kwargs:
            port = 6000
            while _env is None:
                try:
                    kwargs["base_port"] = port
                    _env = default_registry[self.env_id].make(**kwargs)
                except UnityWorkerInUseException:
                    port += 1
                    pass
        else:
            _env = default_registry[self.env_id].make(**kwargs)
        return UnityAECEnv(_env, seed)


# Register each environment in default_registry as a PettingZooEnv
for key in default_registry:
    env_name = key
    if key[0].isdigit():
        env_name = key.replace("3", "Three")
    if not env_name.isidentifier():
        logger.warning(
            f"Environment id {env_name} can not be registered since it is"
            f"not a valid identifier name."
        )
        continue
    locals()[env_name] = PettingZooEnv(key)

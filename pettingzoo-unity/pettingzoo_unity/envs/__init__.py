from mlagents_envs.registry import default_registry
from pettingzoo_unity import UnityToPettingZooWrapper
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

logger = logging_util.get_logger(__name__)
logging_util.set_log_level(logging_util.WARNING)


class petting_zoo_env:
    def __init__(self, env_id):
        self.env_id = env_id

    def env(self, seed: Optional[int] = None, **kwargs):
        if "side_channels" not in kwargs:
            kwargs["side_channels"] = [
                EngineConfigurationChannel(),
                EnvironmentParametersChannel(),
                StatsSideChannel(),
            ]
        _env = None
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
        return UnityToPettingZooWrapper(_env, seed)


for env_id in default_registry:
    env_id = env_id.replace("3DBall", "ThreeDBall")
    if not env_id.isidentifier():
        logger.warning(
            f"Environment id {env_id} can not be registered since it is"
            f"not a valid identifier name."
        )
        continue
    locals()[env_id] = petting_zoo_env(env_id)

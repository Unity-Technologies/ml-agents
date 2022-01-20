from mlagents_envs.registry import default_registry
from mlagents_envs.envs.pettingzoo_env_factory import logger, PettingZooEnvFactory

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
    locals()[env_name] = PettingZooEnvFactory(key)

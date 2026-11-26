from functools import lru_cache as _lru_cache
from importlib.util import find_spec as _find_spec

from mlagents_envs.registry import default_registry
from mlagents_envs.envs.pettingzoo_env_factory import logger, PettingZooEnvFactory


@_lru_cache(maxsize=1)
def _example_factories() -> dict[str, PettingZooEnvFactory]:
    factories = {}
    for key in default_registry:
        env_name = key
        if key and key[0].isdigit():
            env_name = key.replace("3", "Three")
        if not env_name.isidentifier():
            logger.warning(
                f"Environment id {env_name} can not be registered since it is "
                f"not a valid identifier name."
            )
            continue
        factories[env_name] = PettingZooEnvFactory(key)
    return factories


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_example_factories()))


def __getattr__(name: str) -> PettingZooEnvFactory | list[str]:
    if name == "__all__":
        return [attr for attr in __dir__() if not attr.startswith("_")]

    # A from-import probes for attributes before falling back to real submodules.
    if (
        name.isidentifier()
        and not name.startswith("__")
        and _find_spec(f"{__name__}.{name}") is None
    ):
        factory = _example_factories().get(name)
        if factory is not None:
            return factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

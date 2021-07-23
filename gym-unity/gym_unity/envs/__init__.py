from gym.envs.registration import register
from mlagents_envs.registry import default_registry
from gym_unity.envs.gym_env import UnityToGymWrapper
from typing import Any, Dict, List, Optional


def gym_entry_point(env_name):
    def create(action_space_seed: Optional[int] = None, **kwargs):
        _e = default_registry[env_name].make(**kwargs)
        return UnityToGymWrapper(_e, action_space_seed)

    return create

for key in default_registry:
    registry_key = key + "-v0"
    try:
        register(id=registry_key, entry_point=gym_entry_point(key))
    except:
        pass

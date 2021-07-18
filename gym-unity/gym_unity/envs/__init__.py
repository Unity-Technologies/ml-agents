from gym.envs.registration import register
from mlagents_envs.registry import default_registry
from gym_unity.envs.ma_gym import MultiAgentGymWrapper


def gym_entry_point(env_name, **kwargs):
    def create(**kwagrs):
        _e = default_registry[env_name].make(**kwargs)
        return MultiAgentGymWrapper(_e)

    return create

for key in default_registry:
    register(id=key + "-v0", entry_point=gym_entry_point(key))

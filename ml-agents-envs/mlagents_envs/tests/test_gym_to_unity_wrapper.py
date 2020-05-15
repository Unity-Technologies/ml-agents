from mlagents_envs.gym_to_unity_wrapper import GymToUnityWrapper
from mlagents_envs.base_env import ActionType
import gym

import pytest


GYM_ENVS = ["CartPole-v1", "MountainCar-v0"]


@pytest.mark.parametrize("name", GYM_ENVS, ids=GYM_ENVS)
def test_creation(name):
    env = GymToUnityWrapper(gym.make(name), name)
    env.close()


@pytest.mark.parametrize("name", GYM_ENVS, ids=GYM_ENVS)
def test_specs(name):
    gym_env = gym.make(name)
    env = GymToUnityWrapper(gym_env, name)
    assert list(env.behavior_specs.keys()) == [name]
    if isinstance(gym_env.action_space, gym.spaces.Box):
        assert env.behavior_specs[name].action_type == ActionType.CONTINUOUS
    elif isinstance(gym_env.action_space, gym.spaces.Discrete):
        assert env.behavior_specs[name].action_type == ActionType.DISCRETE
    else:
        raise NotImplementedError("Test for this action space type not implemented")
    env.close()


@pytest.mark.parametrize("name", GYM_ENVS, ids=GYM_ENVS)
def test_steps(name):
    env = GymToUnityWrapper(gym.make(name), name)
    spec = env.behavior_specs[name]
    env.reset()
    for _ in range(200):
        d_steps, t_steps = env.get_steps(name)
        env.set_actions(name, spec.create_empty_action(len(d_steps)))
        env.step()
    env.close()

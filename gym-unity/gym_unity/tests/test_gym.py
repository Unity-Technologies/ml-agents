import pytest

import gym
import gym_unity


@pytest.mark.parametrize("env_name", ["3DBall-v0", "WallJump-v0", "GridWorld-v0"])
def test_env(env_name):
    env = gym.make(env_name)
    obs = env.reset()
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    obs = env.reset()
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    assert isinstance(env.action_space, gym.spaces.Space)
    assert isinstance(env.observation_space, gym.spaces.Space)
    env.close()

    env = gym.make(env_name, action_space_seed=42)
    env.reset()
    rng_1 = env.action_space._np_random
    env.close()
    env.action_space

    env = gym.make(env_name, action_space_seed=42)
    env.reset()
    rng_2 = env.action_space._np_random
    env.close()
    assert rng_1.randint(10000000000000) == rng_2.randint(10000000000000)




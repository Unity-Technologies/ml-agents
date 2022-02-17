import pytest

from stable_baselines3 import PPO

from mlagents_envs.envs.unity_vec_env import LimitedConfig, make_mla_sb3_env
from mlagents_envs.registry import default_registry

BASIC_ID = "Basic"


@pytest.mark.parametrize("n_ports", [2])
def test_vec_env_basic(base_port: int) -> None:
    num_envs = 2
    sb3_vec_env = make_mla_sb3_env(
        config=LimitedConfig(
            env_path_or_name=BASIC_ID,
            base_port=base_port,
            num_env=num_envs,
            visual_obs=False,
            allow_multiple_obs=True,
            env_registry=default_registry,
        ),
        # Args to UnityEnvironment
        no_graphics=True,
        num_areas=1,
    )
    assert sb3_vec_env.num_envs == num_envs
    sb3_vec_env.reset()
    observation, reward, done, info = sb3_vec_env.step(
        [sb3_vec_env.action_space.sample()] * 2
    )
    assert len(observation) == num_envs
    assert len(reward) == num_envs
    assert len(done) == num_envs
    assert len(info) == num_envs
    sb3_vec_env.close()


@pytest.mark.slow
@pytest.mark.parametrize("n_ports", [4])
def test_vec_env_trains(base_port: int) -> None:
    sb3_vec_env = make_mla_sb3_env(
        config=LimitedConfig(
            env_path_or_name=BASIC_ID,
            base_port=base_port,
            num_env=4,
            visual_obs=False,
            allow_multiple_obs=True,
            env_registry=default_registry,
        ),
        # Args to UnityEnvironment
        no_graphics=True,
        num_areas=1,
    )

    model = PPO(
        "MlpPolicy",
        sb3_vec_env,
        verbose=1,
        learning_rate=lambda progress: 0.0003 * (1.0 - progress),
    )
    model.learn(total_timesteps=6000)
    sb3_vec_env.close()


# TODO(https://jira.unity3d.com/browse/MLA-2404): Add longer running nightly tests to make sure this trains.

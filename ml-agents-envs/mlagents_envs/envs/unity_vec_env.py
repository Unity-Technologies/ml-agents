from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import gym
from gym import Env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from supersuit import observation_lambda_v0

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.registry import UnityEnvRegistry, default_registry
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig,
    EngineConfigurationChannel,
)

# Default values from CLI (See cli_utils.py)
DEFAULT_ENGINE_CONFIG = EngineConfig(
    width=84,
    height=84,
    quality_level=4,
    time_scale=20,
    target_frame_rate=-1,
    capture_frame_rate=60,
)


# Some config subset of an actual config.yaml file for MLA.
@dataclass
class LimitedConfig:
    # The local path to a Unity executable or the name of an entry in the registry.
    env_path_or_name: str
    base_port: int
    base_seed: int = 0
    num_env: int = 1
    engine_config: EngineConfig = DEFAULT_ENGINE_CONFIG
    visual_obs: bool = False
    # TODO: Decide if we should just tell users to always use MultiInputPolicy so we can simplify the user workflow.
    # WARNING: Make sure to use MultiInputPolicy if you turn this on.
    allow_multiple_obs: bool = False
    env_registry: UnityEnvRegistry = default_registry


def _unity_env_from_path_or_registry(
    env: str, registry: UnityEnvRegistry, **kwargs: Any
) -> UnityEnvironment:
    if Path(env).expanduser().absolute().exists():
        return UnityEnvironment(file_name=env, **kwargs)
    elif env in registry:
        return registry.get(env).make(**kwargs)
    else:
        raise ValueError(f"Environment '{env}' wasn't a local path or registry entry")


def make_mla_sb3_env(config: LimitedConfig, **kwargs: Any) -> VecEnv:
    """
    Create a VecEnv (Stable Baselines 3) of multiple UnityEnvironments.
    :param config: Specifics around initializing the UnityEnvironment.
    :param kwargs: Any other args that need to be passed to the UnityEnvironment constructor that aren't supported
                   through the config.
    :return: A VecEnv backed by Unity environments as specified in the conifg.

    Example Usage:
        # See ml-agents-envs/tests/test_unity_vec_env.py or colab/Colab_UnityEnvironment_4_SB3VectorEnv.ipynb
        sb3_vec_env = make_mla_sb3_env(
            config=LimitedConfig(
                env_path_or_name=BASIC_ID,
                base_port=6000,
                num_env=2,
            ),
            # Other args to UnityEnvironment
            no_graphics=True,
            num_areas=1,
        )
    """

    def handle_obs(obs, space):
        if isinstance(space, gym.spaces.Tuple):
            if len(space) == 1:
                return obs[0]
            # Turn the tuple into a dict (stable baselines can handle spaces.Dict but not spaces.Tuple).
            return {str(i): v for i, v in enumerate(obs)}
        return obs

    def handle_obs_space(space):
        if isinstance(space, gym.spaces.Tuple):
            if len(space) == 1:
                return space[0]
            # Turn the tuple into a dict (stable baselines can handle spaces.Dict but not spaces.Tuple).
            return gym.spaces.Dict({str(i): v for i, v in enumerate(space)})
        return space

    def create_env(env: str, worker_id: int) -> Callable[[], Env]:
        def _f() -> Env:
            engine_configuration_channel = EngineConfigurationChannel()
            engine_configuration_channel.set_configuration(config.engine_config)
            kwargs["side_channels"] = kwargs.get("side_channels", []) + [
                engine_configuration_channel
            ]
            unity_env = _unity_env_from_path_or_registry(
                env=env,
                registry=config.env_registry,
                worker_id=worker_id,
                base_port=config.base_port,
                seed=config.base_seed + worker_id,
                **kwargs,
            )
            new_env = UnityToGymWrapper(
                unity_env=unity_env,
                uint8_visual=config.visual_obs,
                allow_multiple_obs=config.allow_multiple_obs,
            )
            new_env = observation_lambda_v0(new_env, handle_obs, handle_obs_space)
            return new_env

        return _f

    env_facts = [
        create_env(config.env_path_or_name, worker_id=x) for x in range(config.num_env)
    ]
    return SubprocVecEnv(env_facts)

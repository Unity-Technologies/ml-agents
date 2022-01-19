from mlagents_envs.registry import default_registry
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple
import numpy as np

BALL_ID = "3DBall"


def test_set_action_single_agent():
    engine_config_channel = EngineConfigurationChannel()
    env = default_registry[BALL_ID].make(
        base_port=6000,
        worker_id=0,
        no_graphics=True,
        side_channels=[engine_config_channel],
    )
    engine_config_channel.set_configuration_parameters(time_scale=100)
    for _ in range(3):
        env.reset()
        behavior_name = list(env.behavior_specs.keys())[0]
        d, t = env.get_steps(behavior_name)
        for _ in range(50):
            for agent_id in d.agent_id:
                action = np.ones((1, 2))
                action_tuple = ActionTuple()
                action_tuple.add_continuous(action)
                env.set_action_for_agent(behavior_name, agent_id, action_tuple)
            env.step()
            d, t = env.get_steps(behavior_name)
    env.close()


def test_set_action_multi_agent():
    engine_config_channel = EngineConfigurationChannel()
    env = default_registry[BALL_ID].make(
        base_port=6001,
        worker_id=0,
        no_graphics=True,
        side_channels=[engine_config_channel],
    )
    engine_config_channel.set_configuration_parameters(time_scale=100)
    for _ in range(3):
        env.reset()
        behavior_name = list(env.behavior_specs.keys())[0]
        d, t = env.get_steps(behavior_name)
        for _ in range(50):
            action = np.ones((len(d), 2))
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action)
            env.set_actions(behavior_name, action_tuple)
            env.step()
            d, t = env.get_steps(behavior_name)
    env.close()

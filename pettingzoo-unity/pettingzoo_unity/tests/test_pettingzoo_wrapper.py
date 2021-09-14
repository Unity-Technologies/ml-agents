from pettingzoo_unity.tests.simple_test_envs import (
    SimpleEnvironment,
    MultiAgentEnvironment,
)
from pettingzoo.test import api_test
from pettingzoo_unity import UnityToPettingZooWrapper


def test_single_agent():
    unity_env = SimpleEnvironment(["test_single"])
    env = UnityToPettingZooWrapper(unity_env)
    api_test(env, num_cycles=10, verbose_progress=False)


def test_multi_agent():
    unity_env = MultiAgentEnvironment(["test_multi_1", "test_multi_2"], num_agents=2)
    env = UnityToPettingZooWrapper(unity_env)
    api_test(env, num_cycles=10, verbose_progress=False)

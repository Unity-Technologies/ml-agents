import atexit
from typing import Optional, Dict, Any, List, Set
import numpy as np
from gym import error, spaces
from mlagents_envs.base_env import BaseEnv
from pettingzoo import ParallelEnv

from pettingzoo_unity.envs.env_helpers import _unwrap_batch_steps, _agent_id_to_behavior


class UnityParallelEnv(ParallelEnv):
    """
    Unity Parallel (PettingZoo) environment wrapper.
    """

    def __init__(self, env: BaseEnv, seed: Optional[int] = None):
        """
        Initializes a Unity Parallel environment wrapper.

        :param env: The UnityEnvironment that is being wrapped.
        :param seed: The seed for the action spaces of the agents.
        """
        super().__init__()
        atexit.register(self.close)
        self._env = env
        self._assert_loaded()

        self._seed = seed
        self._possible_agents: Set[str] = set()
        self._observations: Dict[str, np.ndarray] = {}
        self._agents: List[str] = []
        self._action_spaces: Dict[str, spaces.Space] = {}
        self._update_action_spaces()

    @property
    def possible_agents(self):
        return self._possible_agents

    @property
    def agents(self):
        return self._agents

    def _assert_loaded(self) -> None:
        if self._env is None:
            raise error.Error("No environment loaded")

    def action_space(self, agent):
        return {
            agent_id: self._action_spaces[_agent_id_to_behavior(agent_id)]
            for agent_id in self._possible_agents
        }

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment.
        """
        self._assert_loaded()
        self._possible_agents = set()
        self._env.reset()
        for behavior_name in self._env.behavior_specs.keys():
            current_batch = self._env.get_steps(behavior_name)
            agents, obs, _, _, _, infos, id_map = _unwrap_batch_steps(
                current_batch, behavior_name
            )
            self._observations.update(obs)
            self._agents += agents

        return self._observations

    def step(self, actions):
        pass

    def render(self, mode="human"):
        pass

    def state(self):
        pass

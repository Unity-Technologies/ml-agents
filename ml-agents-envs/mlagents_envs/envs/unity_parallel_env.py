from typing import Optional, Dict, Any, Tuple
from gym import error
from mlagents_envs.base_env import BaseEnv
from pettingzoo import ParallelEnv

from mlagents_envs.envs.unity_pettingzoo_base_env import UnityPettingzooBaseEnv


class UnityParallelEnv(UnityPettingzooBaseEnv, ParallelEnv):
    """
    Unity Parallel (PettingZoo) environment wrapper.
    """

    def __init__(self, env: BaseEnv, seed: Optional[int] = None):
        """
        Initializes a Unity Parallel environment wrapper.

        :param env: The UnityEnvironment that is being wrapped.
        :param seed: The seed for the action spaces of the agents.
        """
        super().__init__(env, seed)

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment.
        """
        super().reset()

        return self._observations

    def step(self, actions: Dict[str, Any]) -> Tuple:
        self._assert_loaded()
        if len(self._live_agents) <= 0 and actions:
            raise error.Error(
                "You must reset the environment before you can perform a step."
            )

        # Process actions
        for current_agent, action in actions.items():
            self._process_action(current_agent, action)

        # Reset reward
        for k in self._rewards.keys():
            self._rewards[k] = 0

        # Step environment
        self._step()

        # Agent cleanup and sorting
        self._cleanup_agents()
        self._live_agents.sort()  # unnecessary, only for passing API test

        return self._observations, self._rewards, self._dones, self._infos

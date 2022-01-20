from typing import Any, Optional
from gym import error
from mlagents_envs.base_env import BaseEnv
from pettingzoo import AECEnv

from mlagents_envs.envs.unity_pettingzoo_base_env import UnityPettingzooBaseEnv


class UnityAECEnv(UnityPettingzooBaseEnv, AECEnv):
    """
    Unity AEC (PettingZoo) environment wrapper.
    """

    def __init__(self, env: BaseEnv, seed: Optional[int] = None):
        """
        Initializes a Unity AEC environment wrapper.

        :param env: The UnityEnvironment that is being wrapped.
        :param seed: The seed for the action spaces of the agents.
        """
        super().__init__(env, seed)

    def step(self, action: Any) -> None:
        """
        Sets the action of the active agent and get the observation, reward, done
        and info of the next agent.
        :param action: The action for the active agent
        """
        self._assert_loaded()
        if len(self._live_agents) <= 0:
            raise error.Error(
                "You must reset the environment before you can perform a step"
            )

        # Process action
        current_agent = self._agents[self._agent_index]
        self._process_action(current_agent, action)

        self._agent_index += 1
        # Reset reward
        for k in self._rewards.keys():
            self._rewards[k] = 0

        if self._agent_index >= len(self._agents) and self.num_agents > 0:
            # The index is too high, time to set the action for the agents we have
            self._step()
            self._live_agents.sort()  # unnecessary, only for passing API test

    def observe(self, agent_id):
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        return (
            self._observations[agent_id],
            self._cumm_rewards[agent_id],
            self._dones[agent_id],
            self._infos[agent_id],
        )

    def last(self, observe=True):
        """
        returns observation, cumulative reward, done, info for the current agent (specified by self.agent_selection)
        """
        obs, reward, done, info = self.observe(self._agents[self._agent_index])
        return obs if observe else None, reward, done, info

    @property
    def agent_selection(self):
        if not self._live_agents:
            # If we had an agent finish then return that agent even though it isn't alive.
            return self._agents[0]
        return self._agents[self._agent_index]

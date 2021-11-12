from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

from gym import error
from mlagents_envs.base_env import BaseEnv
from pettingzoo import AECEnv

from pettingzoo_unity.envs import UnityBaseEnv
from pettingzoo_unity.envs.env_helpers import _unwrap_batch_steps


def _parse_behavior(full_behavior):
    parsed = urlparse(full_behavior)
    name = parsed.path
    ids = parse_qs(parsed.query)
    team_id: int = 0
    if "team" in ids:
        team_id = int(ids["team"][0])
    return name, team_id


class UnityAECEnv(UnityBaseEnv, AECEnv):
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
            for behavior_name, actions in self._current_action.items():
                self._env.set_actions(behavior_name, actions)
            self._env.step()

            self._reset_states()
            for behavior_name in self._env.behavior_specs.keys():
                current_batch = self._env.get_steps(behavior_name)
                self._current_action[behavior_name] = self._create_empty_actions(
                    behavior_name, len(current_batch[0])
                )
                agents, obs, dones, rewards, cumm_rewards, infos, id_map = _unwrap_batch_steps(
                    current_batch, behavior_name
                )
                self._live_agents += agents
                self._agents += agents
                self._observations.update(obs)
                self._dones.update(dones)
                self._rewards.update(rewards)
                self._cumm_rewards.update(cumm_rewards)
                self._infos.update(infos)
                self._agent_id_to_index.update(id_map)
                self._possible_agents.update(agents)
            self._agent_index = 0
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

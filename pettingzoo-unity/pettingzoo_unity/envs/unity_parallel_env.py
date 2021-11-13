from typing import Optional, Dict, Any
from gym import error
from mlagents_envs.base_env import BaseEnv
from pettingzoo import ParallelEnv
from pettingzoo_unity.envs import UnityBaseEnv
from pettingzoo_unity.envs.env_helpers import _unwrap_batch_steps


class UnityParallelEnv(UnityBaseEnv, ParallelEnv):
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

    def step(self, actions: Dict[str, Any]):
        self._assert_loaded()
        if len(self._live_agents) <= 0:
            raise error.Error(
                "You must reset the environment before you can perform a step."
            )

        # Process actions
        for current_agent, action in actions.items():
            self._process_action(current_agent, action)

        # Reset reward
        for k in self._rewards.keys():
            self._rewards[k] = 0

        # step environment
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
        self._cleanup_agents()
        self._live_agents.sort()  # unnecessary, only for passing API test

        return self._observations, self._rewards, self._dones, self._infos

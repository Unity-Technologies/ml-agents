import atexit
import numpy as np
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from gym import error, spaces

from mlagents_envs.base_env import ActionTuple, BaseEnv

from pettingzoo import AECEnv


def _parse_behavior(full_behavior):
    parsed = urlparse(full_behavior)
    name = parsed.path
    ids = parse_qs(parsed.query)
    team_id: int = 0
    if "team" in ids:
        team_id = int(ids["team"][0])
    return name, team_id


class UnityToPettingZooWrapper(AECEnv):
    def __init__(self, env: BaseEnv, seed: Optional[int] = None):
        """
        :param env: The UnityEnvironment that is being wrapped.
        :param seed: The seed for the action spaces of the agents.
        """
        atexit.register(self.close)
        self._env = env
        self._assert_loaded()

        self._agent_index = 0
        self._behavior_index = 0
        self._seed = seed
        self._side_channel_dict = {
            type(v).__name__: v
            for v in self._env._side_channel_manager._side_channels_dict.values()  # type: ignore
        }

        self._agents: List[str] = []  # agent_id
        self._obs: Dict[str, np.ndarray] = {}  # agent_id: obs
        self._dones: Dict[str, bool] = {}  # agent_id: done
        self._rewards: Dict[str, float] = {}  # agent_id: reward
        self._cumm_rewards: Dict[str, float] = {}  # agent_id: reward
        self._infos: Dict[str, Dict] = {}  # agent_id: info
        self._action_spaces: Dict[str, spaces.Space] = {}  # behavior_name: action_space
        self._obs_spaces: Dict[str, spaces.Space] = {}  # behavior_name: obs_space
        self._current_action: Optional[ActionTuple] = None

    def _assert_loaded(self) -> None:
        if self._env is None:
            raise error.Error("No environment loaded")

    def _current_behavior_name(self) -> str:
        names = list(self._env.behavior_specs.keys())
        return names[self._behavior_index]

    def _behavior2agentid(self, behavior_name: str, index: int) -> str:
        return f"{behavior_name}?agentid={index}"

    def _agentid2behavior(self, agentid: str) -> str:
        return agentid.split("?agentid=")[0]

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """
        Return the observation spaces of all the agents.
        """
        self._assert_loaded()
        for behavior_name in self._env.behavior_specs.keys():
            if behavior_name not in self._obs_spaces:
                obs_spec = self._env.behavior_specs[behavior_name].observation_specs
                obs_spaces = tuple(
                    spaces.Box(-1e7, 1e7, shape=spec.shape, dtype=np.float32)
                    for spec in obs_spec
                )
                if len(obs_spaces) == 1:
                    self._obs_spaces[behavior_name] = obs_spaces[0]
                else:
                    self._obs_spaces[behavior_name] = spaces.Tuple(obs_spaces)
        return {
            agent: self._obs_spaces[self._current_behavior_name()]
            for agent in self._agents
        }

    @property
    def observation_space(self) -> Optional[spaces.Space]:
        """
        The observation space of the current agent.
        """
        self._assert_loaded()
        agent_id = self._agents[self._agent_index]
        behavior_name = self._agentid2behavior(agent_id)
        return self._obs_spaces[behavior_name]

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """
        Return the action spaces of all the agents.
        """
        self._assert_loaded()
        for behavior_name in self._env.behavior_specs.keys():
            if behavior_name not in self._action_spaces:
                act_spec = self._env.behavior_specs[behavior_name].action_spec
                if (
                    act_spec.continuous_size == 0
                    and len(act_spec.discrete_branches) == 0
                ):
                    raise error.Error("No actions found")
                if act_spec.discrete_size == 1:
                    d_space = spaces.Discrete(act_spec.discrete_branches[0])
                    if self._seed is not None:
                        d_space.seed(self._seed)
                    if act_spec.continuous_size == 0:
                        self._action_spaces[behavior_name] = d_space
                        continue
                if act_spec.discrete_size > 0:
                    d_space = spaces.MultiDiscrete(act_spec.discrete_branches)
                    if self._seed is not None:
                        d_space.seed(self._seed)
                    if act_spec.continuous_size == 0:
                        self._action_spaces[behavior_name] = d_space
                        continue
                if act_spec.continuous_size > 0:
                    c_space = spaces.Box(
                        -1, 1, (act_spec.continuous_size,), dtype=np.int32
                    )
                    if self._seed is not None:
                        c_space.seed(self._seed)
                    if len(act_spec.discrete_branches) == 0:
                        self._action_spaces[behavior_name] = c_space
                        continue
                self._action_spaces[behavior_name] = spaces.Tuple((c_space, d_space))
        return {
            agent: self._action_spaces[self._current_behavior_name()]
            for agent in self._agents
        }

    @property
    def action_space(self) -> Optional[spaces.Space]:
        """
        The action space of the current agent.
        """
        self._assert_loaded()
        agent_id = self._agents[self._agent_index]
        behavior_name = self._agentid2behavior(agent_id)
        return self._action_spaces[behavior_name]

    @property
    def side_channel(self) -> Dict[str, Any]:
        """
        The side channels of the environment. You can access the side channels
        of an environment with `env.side_channel[<name-of-channel>]`.
        """
        self._assert_loaded()
        return self._side_channel_dict

    def step(self, action: Any = None) -> None:
        """
        Sets the action of the active agent and get the observation, reward, done
        and info of the next agent.
        :param action: The action for the active agent
        """
        self._assert_loaded()
        if len(self._agents) <= 0:
            raise error.Error(
                "You must reset the environment before you can perform a step"
            )

        # Convert actions
        if action is not None:
            if type(action) != np.ndarray:
                action = np.array(action)
            if not self.action_space.contains(action):  # type: ignore
                raise error.Error(
                    f"Invalid action, got {action} but was expecting action from {self.action_space}"
                )
            if isinstance(self.action_space, spaces.Tuple):
                action = ActionTuple(action[0], action[1])
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                action = ActionTuple(None, action)
            elif isinstance(self.action_space, spaces.Discrete):
                action = ActionTuple(None, np.array(action).reshape(1, 1))
            else:
                action = ActionTuple(action, None)

        # Set action
        if action is not None and self._current_action is not None:
            if (
                self._agent_index >= 0
                and not self._dones[self._agents[self._agent_index]]
            ):
                if action.continuous is not None:
                    self._current_action.continuous[
                        self._agent_index
                    ] = action.continuous[0]
                if action.discrete is not None:
                    self._current_action.discrete[self._agent_index] = action.discrete[
                        0
                    ]
            else:
                # A useless action was passed (the last.agent was done)
                pass

        self._agent_index += 1
        for k in self._rewards.keys():
            self._rewards[k] = 0

        if self._agent_index >= len(self._agents):
            # The index is too high, time to set the action for the agents we have
            if self._current_action is not None:
                self._env.set_actions(
                    self._current_behavior_name(), self._current_action
                )
            self._behavior_index += 1
            if self._behavior_index >= len(self._env.behavior_specs):
                self._env.step()
                self._behavior_index = 0
            current_batch = self._env.get_steps(self._current_behavior_name())
            self._reset_current_action(len(current_batch[0]))
            unwrap = self._unwrap_batch_steps(
                current_batch, self._current_behavior_name()
            )
            self._agents, self._obs, self._dones, self._rewards, self._cumm_rewards, self._infos = (
                unwrap
            )
            self._agent_index = 0

    def _unwrap_batch_steps(self, batch_steps, behavior_name):
        decision_batch, termination_batch = batch_steps
        agents = [
            self._behavior2agentid(behavior_name, i) for i in termination_batch.keys()
        ] + [self._behavior2agentid(behavior_name, i) for i in decision_batch.keys()]
        obs = {
            self._behavior2agentid(behavior_name, i): [
                batch_obs[i] for batch_obs in termination_batch.obs
            ]
            for i in termination_batch.keys()
        }
        obs.update(
            {
                self._behavior2agentid(behavior_name, i): [
                    batch_obs[i] for batch_obs in decision_batch.obs
                ]
                for i in decision_batch.keys()
            }
        )
        obs = {k: v if len(v) > 1 else v[0] for k, v in obs.items()}
        dones = {
            self._behavior2agentid(behavior_name, i): True
            for i in termination_batch.keys()
        }
        dones.update(
            {
                self._behavior2agentid(behavior_name, i): False
                for i in decision_batch.keys()
            }
        )
        rewards = {
            self._behavior2agentid(behavior_name, i): termination_batch.reward[i]
            for i in termination_batch.keys()
        }
        rewards.update(
            {
                self._behavior2agentid(behavior_name, i): decision_batch.reward[i]
                for i in decision_batch.keys()
            }
        )
        cumm_rewards = {k: v for k, v in rewards.items()}
        infos = {x: {} for x in agents}
        return agents, obs, dones, rewards, cumm_rewards, infos

    def _reset_current_action(self, num_agents):
        a_spec = self._env.behavior_specs[self._current_behavior_name()].action_spec
        self._current_action = ActionTuple(
            np.zeros((num_agents, a_spec.continuous_size), dtype=np.float32),
            np.zeros((num_agents, len(a_spec.discrete_branches)), dtype=np.int32),
        )

    def reset(self):
        """
        Resets the environment.
        """
        self._assert_loaded()
        self._agent_index = 0
        self._behavior_index = 0
        self._env.reset()
        current_batch = self._env.get_steps(self._current_behavior_name())
        self._agents, self._obs, _, _, _, _ = self._unwrap_batch_steps(
            current_batch, self._current_behavior_name()
        )
        self._dones = {agent: False for agent in self._agents}
        self._rewards = {agent: 0 for agent in self._agents}
        self._cumm_rewards = {agent: 0 for agent in self._agents}
        self._infos = {agent: {} for agent in self._agents}
        self._reset_current_action(len(current_batch[0]))  # len of decision steps
        self.step()

    def seed(self, seed=None):
        """
        Reseeds the environment (making the resulting environment deterministic).
        `reset()` must be called after `seed()`, and before `step()`.
        """
        self._seed = seed

    def render(self, mode="human"):
        """
        NOT SUPPORTED.

        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside of classic,
        and `'ansi'` which returns the strings printed (specific to classic environments).
        """
        pass

    def observe(self, agent):
        """
        Returns the observation an agent currently can make. `last()` calls this function.
        """
        agent_id = self._behavior2agentid(self._current_behavior_name(), agent)
        return (
            self._obs[agent_id],
            self._cumm_rewards[agent_id],
            self._dones[agent_id],
            {},
        )

    def last(self, observe=True):
        """
        returns observation, cumulative reward, done, info for the current agent (specified by self.agent_selection)
        """
        obs, reward, done, info = self.observe(self._agent_index)
        return obs if observe else None, reward, done, info

    @property
    def dones(self):
        return dict(self._dones)

    @property
    def agents(self):
        return self._agents

    @property
    def rewards(self):
        return dict(self._rewards)

    @property
    def infos(self):
        return dict(self._infos)

    @property
    def agent_selection(self):
        return self._agents[self._agent_index]

    @property
    def possible_agents(self):
        return list(self._agents)

    def close(self) -> None:
        """
        Close the environment.
        """
        if self._env is not None:
            self._env.close()
            self._env = None  # type: ignore

    def __del__(self) -> None:
        self.close()


if __name__ == "__main__":
    # run pettingzoo api test
    from pettingzoo.test import api_test
    from mlagents_envs.registry import default_registry

    unity_env = default_registry["3DBall"].make()
    env = UnityToPettingZooWrapper(unity_env)
    api_test(env, num_cycles=10, verbose_progress=False)

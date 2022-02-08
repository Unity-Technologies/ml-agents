import atexit
from typing import Optional, List, Set, Dict, Any, Tuple
import numpy as np
from gym import error, spaces
from mlagents_envs.base_env import BaseEnv, ActionTuple
from mlagents_envs.envs.env_helpers import _agent_id_to_behavior, _unwrap_batch_steps


class UnityPettingzooBaseEnv:
    """
    Unity Petting Zoo base environment.
    """

    def __init__(
        self, env: BaseEnv, seed: Optional[int] = None, metadata: Optional[dict] = None
    ):
        super().__init__()
        atexit.register(self.close)
        self._env = env
        self.metadata = metadata
        self._assert_loaded()

        self._agent_index = 0
        self._seed = seed
        self._side_channel_dict = {
            type(v).__name__: v
            for v in self._env._side_channel_manager._side_channels_dict.values()  # type: ignore
        }

        self._live_agents: List[str] = []  # agent id for agents alive
        self._agents: List[str] = []  # all agent id in current step
        self._possible_agents: Set[str] = set()  # all agents that have ever appear
        self._agent_id_to_index: Dict[str, int] = {}  # agent_id: index in decision step
        self._observations: Dict[str, np.ndarray] = {}  # agent_id: obs
        self._dones: Dict[str, bool] = {}  # agent_id: done
        self._rewards: Dict[str, float] = {}  # agent_id: reward
        self._cumm_rewards: Dict[str, float] = {}  # agent_id: reward
        self._infos: Dict[str, Dict] = {}  # agent_id: info
        self._action_spaces: Dict[str, spaces.Space] = {}  # behavior_name: action_space
        self._observation_spaces: Dict[
            str, spaces.Space
        ] = {}  # behavior_name: obs_space
        self._current_action: Dict[str, ActionTuple] = {}  # behavior_name: ActionTuple
        # Take a single step so that the brain information will be sent over
        if not self._env.behavior_specs:
            self._env.step()
            for behavior_name in self._env.behavior_specs.keys():
                _, _, _ = self._batch_update(behavior_name)
        self._update_observation_spaces()
        self._update_action_spaces()

    def _assert_loaded(self) -> None:
        if self._env is None:
            raise error.Error("No environment loaded")

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """
        Return the observation spaces of all the agents.
        """
        return {
            agent_id: self._observation_spaces[_agent_id_to_behavior(agent_id)]
            for agent_id in self._possible_agents
        }

    def observation_space(self, agent: str) -> Optional[spaces.Space]:
        """
        The observation space of the current agent.
        """
        behavior_name = _agent_id_to_behavior(agent)
        return self._observation_spaces[behavior_name]

    def _update_observation_spaces(self) -> None:
        self._assert_loaded()
        for behavior_name in self._env.behavior_specs.keys():
            if behavior_name not in self._observation_spaces:
                obs_spec = self._env.behavior_specs[behavior_name].observation_specs
                obs_spaces = tuple(
                    spaces.Box(
                        low=-np.float32(np.inf),
                        high=np.float32(np.inf),
                        shape=spec.shape,
                        dtype=np.float32,
                    )
                    for spec in obs_spec
                )
                if len(obs_spaces) == 1:
                    self._observation_spaces[behavior_name] = obs_spaces[0]
                else:
                    self._observation_spaces[behavior_name] = spaces.Tuple(obs_spaces)

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """
        Return the action spaces of all the agents.
        """
        return {
            agent_id: self._action_spaces[_agent_id_to_behavior(agent_id)]
            for agent_id in self._possible_agents
        }

    def action_space(self, agent: str) -> Optional[spaces.Space]:
        """
        The action space of the current agent.
        """
        behavior_name = _agent_id_to_behavior(agent)
        return self._action_spaces[behavior_name]

    def _update_action_spaces(self) -> None:
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

    def _process_action(self, current_agent, action):
        current_action_space = self.action_space(current_agent)
        # Convert actions
        if action is not None:
            if isinstance(action, Tuple):
                action = tuple(np.array(a) for a in action)
            else:
                action = self._action_to_np(current_action_space, action)
            if not current_action_space.contains(action):  # type: ignore
                raise error.Error(
                    f"Invalid action, got {action} but was expecting action from {self.action_space}"
                )
            if isinstance(current_action_space, spaces.Tuple):
                action = ActionTuple(action[0], action[1])
            elif isinstance(current_action_space, spaces.MultiDiscrete):
                action = ActionTuple(None, action)
            elif isinstance(current_action_space, spaces.Discrete):
                action = ActionTuple(None, np.array(action).reshape(1, 1))
            else:
                action = ActionTuple(action, None)

        if not self._dones[current_agent]:
            current_behavior = _agent_id_to_behavior(current_agent)
            current_index = self._agent_id_to_index[current_agent]
            if action.continuous is not None:
                self._current_action[current_behavior].continuous[
                    current_index
                ] = action.continuous[0]
            if action.discrete is not None:
                self._current_action[current_behavior].discrete[
                    current_index
                ] = action.discrete[0]
        else:
            self._live_agents.remove(current_agent)
            del self._observations[current_agent]
            del self._dones[current_agent]
            del self._rewards[current_agent]
            del self._cumm_rewards[current_agent]
            del self._infos[current_agent]

    def _step(self):
        for behavior_name, actions in self._current_action.items():
            self._env.set_actions(behavior_name, actions)
        self._env.step()
        self._reset_states()
        for behavior_name in self._env.behavior_specs.keys():
            dones, rewards, cumulative_rewards = self._batch_update(behavior_name)
            self._dones.update(dones)
            self._rewards.update(rewards)
            self._cumm_rewards.update(cumulative_rewards)
        self._agent_index = 0

    def _cleanup_agents(self):
        for current_agent, done in self.dones.items():
            if done:
                self._live_agents.remove(current_agent)

    @property
    def side_channel(self) -> Dict[str, Any]:
        """
        The side channels of the environment. You can access the side channels
        of an environment with `env.side_channel[<name-of-channel>]`.
        """
        self._assert_loaded()
        return self._side_channel_dict

    @staticmethod
    def _action_to_np(current_action_space, action):
        return np.array(action, dtype=current_action_space.dtype)

    def _create_empty_actions(self, behavior_name, num_agents):
        a_spec = self._env.behavior_specs[behavior_name].action_spec
        return ActionTuple(
            np.zeros((num_agents, a_spec.continuous_size), dtype=np.float32),
            np.zeros((num_agents, len(a_spec.discrete_branches)), dtype=np.int32),
        )

    @property
    def _cumulative_rewards(self):
        return self._cumm_rewards

    def _reset_states(self):
        self._live_agents = []
        self._agents = []
        self._observations = {}
        self._dones = {}
        self._rewards = {}
        self._cumm_rewards = {}
        self._infos = {}
        self._agent_id_to_index = {}

    def reset(self):
        """
        Resets the environment.
        """
        self._assert_loaded()
        self._agent_index = 0
        self._reset_states()
        self._possible_agents = set()
        self._env.reset()
        for behavior_name in self._env.behavior_specs.keys():
            _, _, _ = self._batch_update(behavior_name)
        self._live_agents.sort()  # unnecessary, only for passing API test
        self._dones = {agent: False for agent in self._agents}
        self._rewards = {agent: 0 for agent in self._agents}
        self._cumm_rewards = {agent: 0 for agent in self._agents}

    def _batch_update(self, behavior_name):
        current_batch = self._env.get_steps(behavior_name)
        self._current_action[behavior_name] = self._create_empty_actions(
            behavior_name, len(current_batch[0])
        )
        (
            agents,
            obs,
            dones,
            rewards,
            cumulative_rewards,
            infos,
            id_map,
        ) = _unwrap_batch_steps(current_batch, behavior_name)
        self._live_agents += agents
        self._agents += agents
        self._observations.update(obs)
        self._infos.update(infos)
        self._agent_id_to_index.update(id_map)
        self._possible_agents.update(agents)
        return dones, rewards, cumulative_rewards

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

    @property
    def dones(self):
        return dict(self._dones)

    @property
    def agents(self):
        return sorted(self._live_agents)

    @property
    def rewards(self):
        return dict(self._rewards)

    @property
    def infos(self):
        return dict(self._infos)

    @property
    def possible_agents(self):
        return sorted(self._possible_agents)

    def close(self) -> None:
        """
        Close the environment.
        """
        if self._env is not None:
            self._env.close()
            self._env = None  # type: ignore

    def __del__(self) -> None:
        self.close()

    def state(self):
        pass

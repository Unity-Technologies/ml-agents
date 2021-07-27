import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import atexit
import gym
from gym import error, spaces
from urllib.parse import urlparse, parse_qs

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps


def _parse_behavior(full_behavior):
    parsed = urlparse(full_behavior)
    name = parsed.path
    ids = parse_qs(parsed.query)
    team_id: int = 0
    if "team" in ids:
        team_id = int(ids["team"][0])
    return name, team_id


class AgentStatus:
    def __init__(
        self,
        behavior,
        agent,
        group,
        group_reward,
        action_mask,
        interrupted,
        current_obs,
        reward,
        done,
        needs_action,
    ):
        self.behavior, self.team = _parse_behavior(behavior)
        self.agent = agent
        self.group = group
        self.group_reward = group_reward
        self.action_mask = action_mask
        self.interrupted = interrupted
        self.obs = current_obs
        self.reward = reward
        self.done = done
        self.needs_action = needs_action


class UnityToGymWrapper(gym.Env):
    def __init__(self, env: BaseEnv, action_space_seed: Optional[int] = None):
        atexit.register(self.close)
        self._env = env
        self._agent_index = -1
        self._current_batch: Optional[Tuple[DecisionSteps, TerminalSteps]] = None
        self._behavior_index = 0
        self._current_action: Optional[ActionTuple] = None
        self._last: Optional[AgentStatus] = None
        self._action_spaces: Dict[str, spaces.Space] = {}
        self._obs_spaces: Dict[str, spaces.Space] = {}
        self._action_sampling_seed = action_space_seed
        self._side_channel_dict = {
            type(v).__name__: v
            for v in self._env._side_channel_manager._side_channels_dict.values()
        }

    def _current_behavior_name(self) -> Optional[str]:
        names = list(self._env.behavior_specs.keys())
        if len(names) > self._behavior_index:
            return names[self._behavior_index]
        return None

    def _assert_loaded(self) -> None:
        if self._env is None:
            raise error.Error("No environment loaded")

    @property
    def active(self) -> Optional[AgentStatus]:
        """
        This is not part of the original gym api. It is needed to know which behaviors
        and agent ID the observation from a reset call belongs to.
        """
        # TODO : info won't do. Info is supposed to be "extra" information, not needed
        self._assert_loaded()
        return self._last

    @property
    def observation_space(self) -> Optional[spaces.Space]:
        self._assert_loaded()
        current_behavior = self._current_behavior_name()
        if current_behavior is None:
            return None
        if current_behavior in self._obs_spaces:
            return self._obs_spaces[current_behavior]
        obs_spec = self._env.behavior_specs[current_behavior].observation_specs
        obs_spaces = tuple(
            spaces.Box(-np.inf, np.inf, shape=spec.shape, dtype=np.float32)
            for spec in obs_spec
        )
        if len(obs_spaces) == 1:
            self._obs_spaces[current_behavior] = obs_spaces[0]
        else:
            self._obs_spaces[current_behavior] = spaces.Tuple(obs_spaces)
        return self._obs_spaces[current_behavior]

    @property
    def action_space(self) -> Optional[spaces.Space]:
        self._assert_loaded()
        current_behavior = self._current_behavior_name()
        if current_behavior is None:
            return None
        if current_behavior in self._action_spaces:
            return self._action_spaces[current_behavior]
        act_spec = self._env.behavior_specs[current_behavior].action_spec
        if act_spec.continuous_size == 0 and len(act_spec.discrete_branches) == 0:
            raise error.Error("No actions found")
        if act_spec.discrete_size == 1:
            d_space = spaces.Discrete(act_spec.discrete_branches[0])
            if self._action_sampling_seed is not None:
                d_space.seed(self._action_sampling_seed)
            if act_spec.continuous_size == 0:
                self._action_spaces[current_behavior] = d_space
                return self._action_spaces[current_behavior]
        if act_spec.discrete_size > 0:
            d_space = spaces.MultiDiscrete(act_spec.discrete_branches)
            if self._action_sampling_seed is not None:
                d_space.seed(self._action_sampling_seed)
            if act_spec.continuous_size == 0:
                self._action_spaces[current_behavior] = d_space
                return self._action_spaces[current_behavior]
        if act_spec.continuous_size > 0:
            c_space = spaces.Box(-1, 1, (act_spec.continuous_size,), dtype=np.int32)
            if self._action_sampling_seed is not None:
                c_space.seed(self._action_sampling_seed)
            if len(act_spec.discrete_branches) == 0:
                self._action_spaces[current_behavior] = c_space
                return self._action_spaces[current_behavior]
        self._action_spaces[current_behavior] = spaces.Tuple((c_space, d_space))
        return self._action_spaces[current_behavior]

    @property
    def reward_range(self) -> Tuple[float, float]:
        self._assert_loaded()
        return -float("inf"), float("inf")

    @property
    def side_channel(self) -> Dict[str, Any]:
        self._assert_loaded()
        return self._side_channel_dict

    def step(
        self, action: Any = None
    ) -> Tuple[List[np.array], float, bool, Optional[Dict[str, Any]]]:
        self._assert_loaded()
        # Convert Actions
        if action is not None:
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

        if self._current_batch is None:
            raise error.Error(
                "You must reset the environment before you can perform a step"
            )

        decision_batch, termination_batch = self._current_batch

        # Build action
        if self._current_action is None:
            a_spec = self._env.behavior_specs[self._current_behavior_name()].action_spec
            self._current_action = ActionTuple(
                np.zeros(
                    (len(decision_batch), a_spec.continuous_size), dtype=np.float32
                ),
                np.zeros(
                    (len(decision_batch), len(a_spec.discrete_branches)), dtype=np.int32
                ),
            )
        if action is not None:
            if self._agent_index >= 0 and self._agent_index < len(decision_batch):

                if action.continuous is not None:
                    self._current_action.continuous[
                        self._agent_index
                    ] = action.continuous[0]
                if action.discrete is not None:
                    self._current_action.discrete[self._agent_index] = action.discrete[
                        0
                    ]
            else:
                # A Useless action was passed (the last.agent was done)
                pass

        self._agent_index += 1

        if self._agent_index < len(termination_batch):
            # The index is within the terminal steps
            index = self._agent_index
            obs = [batch_obs[index] for batch_obs in termination_batch.obs]
            obs = obs if len(obs) > 1 else obs[0]
            reward = termination_batch.reward[index]
            done = True
            self._last = AgentStatus(
                list(self._env.behavior_specs.keys())[self._behavior_index],
                termination_batch.agent_id[index],
                termination_batch.group_id[index],
                termination_batch.group_reward[index],
                None,
                termination_batch.interrupted[index],
                obs,
                reward,
                done,
                False,
            )
            return obs, reward, done, None

        if self._agent_index < len(decision_batch) + len(termination_batch):
            index = self._agent_index - len(termination_batch)
            # the index is within the decsion steps
            obs = [batch_obs[index] for batch_obs in decision_batch.obs]
            obs = obs if len(obs) > 1 else obs[0]
            reward = decision_batch.reward[index]
            done = False

            self._last = AgentStatus(
                list(self._env.behavior_specs.keys())[self._behavior_index],
                decision_batch.agent_id[index],
                decision_batch.group_id[index],
                decision_batch.group_reward[index],
                [mask[index] for mask in decision_batch.action_mask]
                if decision_batch.action_mask is not None
                else None,
                False,
                obs,
                reward,
                done,
                True,
            )
            return obs, reward, done, None

        # The index is too high, time to set the action for the agents we have
        if self._current_action is not None:
            self._env.set_actions(self._current_behavior_name(), self._current_action)
        self._current_action = None
        self._behavior_index += 1
        if self._behavior_index >= len(self._env.behavior_specs):
            self._env.step()
            self._behavior_index = 0
        self._current_batch = self._env.get_steps(self._current_behavior_name())
        self._agent_index = -1

        return self.step()

    def reset(self) -> List[np.array]:
        self._assert_loaded()
        self._env.reset()
        self._agent_index = -1
        self._behavior_index = 0
        self._current_batch = self._env.get_steps(self._current_behavior_name())
        self._current_action = None
        obs, _, _, _ = self.step()
        return obs

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        self.close()

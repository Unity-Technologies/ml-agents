import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
from gym import error, spaces
from urllib.parse import urlparse, parse_qs

from mlagents_envs.base_env import ActionTuple, BaseEnv
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.base_env import BaseEnv


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


class MultiAgentGymWrapper(gym.Env):
    def __init__(self, env: BaseEnv):
        self._env = env
        self._agent_index = -1
        self._current_batch: Optional[Tuple[DecisionSteps, TerminalSteps]] = None
        self._behavior_index = 0
        self._current_action: Optional[ActionTuple] = None
        self._last: Optional[AgentStatus] = None

    def _current_behavior_name(self) -> str:
        return list(self._env.behavior_specs.keys())[self._behavior_index]

    @property
    def last(self) -> Optional[AgentStatus]:
        """
        This is not part of the original gym api. It is needed to know which behaviors
        and agent ID the observation from a reset call belongs to.
        """
        # TODO : info won't do. Info is supposed to be "extra" information, not needed
        return self._last

    @property
    def observation_space(self) -> spaces.Space:
        current_behavior = self._current_behavior_name()
        obs_spec = self._env.behavior_specs[current_behavior].observation_specs
        obs_spaces = tuple(
            [
                spaces.Box(-np.inf, np.inf, shape=spec.shape, dtype=np.float32)
                for spec in obs_spec
            ]
        )
        return spaces.Tuple(obs_spaces)

    @property
    def action_space(self) -> spaces.Space:
        current_behavior = self._current_behavior_name()
        act_spec = self._env.behavior_specs[current_behavior].action_spec
        if act_spec.continuous_size == 0 and len(act_spec.discrete_branches) == 0:
            raise Exception("No actions found")
        if act_spec.discrete_size == 1:
            d_space = spaces.Discrete(act_spec.discrete_branches[0])
            if act_spec.continuous_size == 0:
                return d_space
        if act_spec.discrete_size > 0:
            d_space = spaces.MultiDiscrete(act_spec.discrete_branches)
            if act_spec.continuous_size == 0:
                return d_space
        if act_spec.continuous_size > 0:
            c_space = spaces.Box(-1, 1, (act_spec.continuous_size,), dtype=np.int32)
            if len(act_spec.discrete_branches) == 0:
                return c_space
        return spaces.Tuple(tuple(c_space, d_space))

    def step(
        self, action: Any = None
    ) -> Tuple[List[np.array], float, bool, Dict[str, Any]]:
        # get one agent at a time. In single agent envs, behaves just like Gym. No need for gym wrapper
        # return List[np.array] (observation), float (reward), bool (done), info (dict with group reward, group_id, behavior_name, agent_id, etc...)

        # Convert Actions
        if action is not None:
            if not self.action_space.contains(action):
                raise Exception(
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
            raise Exception(
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

        if self._agent_index < len(decision_batch):

            # the index is within the decsion steps
            obs = [batch_obs[self._agent_index] for batch_obs in decision_batch.obs]
            reward = decision_batch.reward[self._agent_index]
            done = False

            self._last = AgentStatus(
                list(self._env.behavior_specs.keys())[self._behavior_index],
                decision_batch.agent_id[self._agent_index],
                decision_batch.group_id[self._agent_index],
                decision_batch.group_reward[self._agent_index],
                [mask[self._agent_index] for mask in decision_batch.action_mask]
                if decision_batch.action_mask is not None
                else None,
                False,
                obs,
                reward,
                done,
                True,
            )

            return obs, reward, done, None

        if self._agent_index < len(decision_batch) + len(termination_batch):
            # The index is within the terminal steps
            index = self._agent_index - len(decision_batch)
            obs = [batch_obs[index] for batch_obs in termination_batch.obs]
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
        self._env.reset()
        self._agent_index = -1
        self._behavior_index = 0
        self._current_batch = self._env.get_steps(self._current_behavior_name())
        self._current_action = None
        obs, _, _, _ = self.step()
        return obs

    def close(self) -> None:
        self._env.close()

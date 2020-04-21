from mlagents_envs.base_env import (
    BaseEnv,
    DecisionSteps,
    TerminalSteps,
    BehaviorSpec,
    BehaviorName,
    AgentId,
    ActionType,
)
from mlagents_envs.exception import UnityActionException, UnityObservationException

from typing import List, Tuple

import numpy as np

try:
    import gym

    _GYM_IMPORTED = True
except ImportError:
    _GYM_IMPORTED = False


class GymToUnityWrapper(BaseEnv):
    _DEFAULT_BEHAVIOR_NAME = "gym_behavior_name"
    _AGENT_ID = 1

    def __init__(self, gym_env, name=None):
        if not _GYM_IMPORTED:
            raise RuntimeError(
                "gym is not installed, gym required to use the GymToUnityWrapper"
            )
        self._gym_env = gym_env
        self._first_message = True
        self._behavior_name = name
        if self._behavior_name is None:
            self._behavior_name = self._DEFAULT_BEHAVIOR_NAME
        action_type = ActionType.CONTINUOUS
        action_shape = 0
        if isinstance(self._gym_env.action_space, gym.spaces.Box):
            action_type = ActionType.CONTINUOUS
            action_shape = np.prod(self._gym_env.action_space.shape)
        elif isinstance(self._gym_env.action_space, gym.spaces.Discrete):
            action_shape = (self._gym_env.action_space.n,)
            action_type = ActionType.DISCRETE
        else:
            raise UnityActionException(
                f"Unknown action type {self._gym_env.action_space}"
            )
        self.obs_ratio = np.maximum(
            self._gym_env.observation_space.high, -self._gym_env.observation_space.low
        )
        if not isinstance(self._gym_env.observation_space, gym.spaces.Box):
            raise UnityObservationException(
                f"Unknown observation type {self._gym_env.observation_space}"
            )
        self._behavior_specs = BehaviorSpec(
            observation_shapes=[self._gym_env.observation_space.shape],
            action_type=action_type,
            action_shape=action_shape,
        )
        self._g_action: np.ndarray = None
        self._current_steps: Tuple[DecisionSteps, TerminalSteps] = (None, None)

    def step(self) -> None:
        if self._first_message:
            self.reset()
            return
        obs, rew, done, info = self._gym_env.step(self._g_action)
        if not done:
            self._current_steps = (
                DecisionSteps(
                    obs=[np.expand_dims(obs / self.obs_ratio, axis=0)],
                    reward=np.array([rew], dtype=np.float32),
                    agent_id=np.array([self._AGENT_ID], dtype=np.int32),
                    action_mask=None,
                ),
                TerminalSteps.empty(self._behavior_specs),
            )
        else:
            self._first_message = True
            self._current_steps = (
                DecisionSteps.empty(self._behavior_specs),
                TerminalSteps(
                    obs=[np.expand_dims(obs / self.obs_ratio, axis=0)],
                    reward=np.array([rew], dtype=np.float32),
                    max_step=np.array(
                        [info.get("TimeLimit.truncated", False)], dtype=np.bool
                    ),
                    agent_id=np.array([self._AGENT_ID], dtype=np.int32),
                ),
            )

    def reset(self) -> None:
        self._first_message = False
        obs = self._gym_env.reset()
        self._current_steps = (
            DecisionSteps(
                obs=[np.expand_dims(obs / self.obs_ratio, axis=0)],
                reward=np.array([0], dtype=np.float32),
                agent_id=np.array([self._AGENT_ID], dtype=np.int32),
                action_mask=None,
            ),
            TerminalSteps.empty(self._behavior_specs),
        )

    def close(self) -> None:
        self._gym_env.close()

    def get_behavior_names(self) -> List[BehaviorName]:
        return [self._behavior_name]

    def set_actions(self, behavior_name: BehaviorName, action: np.ndarray) -> None:
        assert behavior_name == self._behavior_name
        spec = self._behavior_specs
        expected_type = np.float32 if spec.is_action_continuous() else np.int32
        n_agents = len(self._current_steps[0])
        expected_shape = (n_agents, spec.action_size)
        if action.shape != expected_shape:
            raise UnityActionException(
                "The behavior {0} needs an input of dimension {1} but received input of dimension {2}".format(
                    behavior_name, expected_shape, action.shape
                )
            )
        if action.dtype != expected_type:
            action = action.astype(expected_type)
        if n_agents == 0:
            return
        if isinstance(self._gym_env.action_space, gym.spaces.Discrete):
            self._g_action = int(action[0, 0])
        elif isinstance(self._gym_env.action_space, gym.spaces.Box):
            self._g_action = action[0]
        else:
            raise UnityActionException(
                f"Unknown action type {self._gym_env.action_space}"
            )

    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: np.ndarray
    ) -> None:
        assert behavior_name == self._behavior_name
        assert agent_id == self._AGENT_ID
        spec = self._behavior_specs
        expected_shape = (spec.action_size,)
        if action.shape != expected_shape:
            raise UnityActionException(
                f"The Agent {0} with BehaviorName {1} needs an input of dimension "
                f"{2} but received input of dimension {3}".format(
                    agent_id, behavior_name, expected_shape, action.shape
                )
            )
        expected_type = np.float32 if spec.is_action_continuous() else np.int32
        if action.dtype != expected_type:
            action = action.astype(expected_type)
        if isinstance(self._gym_env.action_space, gym.spaces.Discrete):
            self._g_action = int(action[0])
        elif isinstance(self._gym_env.action_space, gym.spaces.Box):
            self._g_action = action
        else:
            raise UnityActionException(
                f"Unknown action type {self._gym_env.action_space}"
            )

    def get_steps(
        self, behavior_name: BehaviorName
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        assert behavior_name == self._behavior_name
        return self._current_steps

    def get_behavior_spec(self, behavior_name: BehaviorName) -> BehaviorSpec:
        assert behavior_name == self._behavior_name
        return self._behavior_specs

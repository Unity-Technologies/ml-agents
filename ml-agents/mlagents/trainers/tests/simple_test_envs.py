import random
from typing import Dict, List
import numpy as np

from mlagents_envs.base_env import (
    BaseEnv,
    AgentGroupSpec,
    BatchedStepResult,
    ActionType,
)

OBS_SIZE = 1
STEP_SIZE = 0.1

TIME_PENALTY = 0.001
MIN_STEPS = int(1.0 / STEP_SIZE) + 1
SUCCESS_REWARD = 1.0 + MIN_STEPS * TIME_PENALTY


def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))


class Simple1DEnvironment(BaseEnv):
    """
    Very simple "game" - the agent has a position on [-1, 1], gets a reward of 1 if it reaches 1, and a reward of -1 if
    it reaches -1. The position is incremented by the action amount (clamped to [-step_size, step_size]).
    """

    def __init__(self, brain_names, use_discrete, step_size=STEP_SIZE):
        super().__init__()
        self.discrete = use_discrete
        action_type = ActionType.DISCRETE if use_discrete else ActionType.CONTINUOUS
        self.group_spec = AgentGroupSpec(
            [(OBS_SIZE,)], action_type, (2,) if use_discrete else 1
        )
        self.names = brain_names
        self.position: Dict[str, float] = {}
        self.step_count: Dict[str, float] = {}
        self.random = random.Random(str(self.group_spec))
        self.goal: Dict[str, int] = {}
        self.action = {}
        self.rewards: Dict[str, float] = {}
        self.final_rewards: Dict[str, List[float]] = {}
        self.step_result: Dict[str, BatchedStepResult] = {}
        self.step_size = step_size  # defines the difficulty of the test

        for name in self.names:
            self.goal[name] = self.random.choice([-1, 1])
            self.rewards[name] = 0
            self.final_rewards[name] = []
            self._reset_agent(name)
            self.action[name] = None
            self.step_result[name] = None

    def get_agent_groups(self):
        return self.names

    def get_agent_group_spec(self, name):
        return self.group_spec

    def set_action_for_agent(self, name, id, data):
        pass

    def set_actions(self, name, data):
        self.action[name] = data

    def get_step_result(self, name):
        return self.step_result[name]

    def _take_action(self, name: str) -> bool:
        if self.discrete:
            act = self.action[name][0][0]
            delta = 1 if act else -1
        else:
            delta = self.action[name][0][0]
        delta = clamp(delta, -self.step_size, self.step_size)
        self.position[name] += delta
        self.position[name] = clamp(self.position[name], -1, 1)
        self.step_count[name] += 1
        done = self.position[name] >= 1.0 or self.position[name] <= -1.0
        return done

    def _compute_reward(self, name: str, done: bool) -> float:
        if done:
            reward = SUCCESS_REWARD * self.position[name] * self.goal[name]
        else:
            reward = -TIME_PENALTY
        return reward

    def _make_batched_step(
        self, name: str, done: bool, reward: float
    ) -> BatchedStepResult:
        m_vector_obs = [np.ones((1, OBS_SIZE), dtype=np.float32) * self.goal[name]]
        m_reward = np.array([reward], dtype=np.float32)
        m_done = np.array([done], dtype=np.bool)
        m_agent_id = np.array([0], dtype=np.int32)
        action_mask = self._generate_mask()
        return BatchedStepResult(
            m_vector_obs, m_reward, m_done, m_done, m_agent_id, action_mask
        )

    def step(self) -> None:
        assert all(action is not None for action in self.action.values())

        for name in self.names:
            done = self._take_action(name)

            reward = self._compute_reward(name, done)
            self.rewards[name] += reward
            self.step_result[name] = self._make_batched_step(name, done, reward)
            if done:
                self._reset_agent(name)

    def _generate_mask(self):
        if self.discrete:
            # LL-Python API will return an empty dim if there is only 1 agent.
            ndmask = np.array(2 * [False], dtype=np.bool)
            ndmask = np.expand_dims(ndmask, axis=0)
            action_mask = [ndmask]
        else:
            action_mask = None
        return action_mask

    def _reset_agent(self, name):
        self.goal[name] = self.random.choice([-1, 1])
        self.position[name] = 0.0
        self.step_count[name] = 0
        self.final_rewards[name].append(self.rewards[name])
        self.rewards[name] = 0

    def reset(self) -> None:  # type: ignore
        for name in self.names:
            self._reset_agent(name)
            self.step_result[name] = self._make_batched_step(name, False, 0.0)

    @property
    def reset_parameters(self) -> Dict[str, str]:
        return {}

    def close(self):
        pass


class Memory1DEnvironment(Simple1DEnvironment):
    def __init__(self, brain_names, use_discrete):
        super().__init__(brain_names, use_discrete)
        action_type = ActionType.DISCRETE if use_discrete else ActionType.CONTINUOUS
        self.group_spec = AgentGroupSpec(
            [(2 * OBS_SIZE,)], action_type, (2,) if use_discrete else 1
        )

    def _reset_agent(self, name: str) -> None:
        super()._reset_agent(name)
        self.goal = self.random.choice([-1, 1])

    def _make_batched_step(
        self, name: str, done: bool, reward: float
    ) -> BatchedStepResult:
        base_obs = np.zeros((1, OBS_SIZE), dtype=np.float32)
        # Feed the recurrent obs for the 1st step
        recurrent_obs_val = self.goal if self.step_count[name] <= 1 else 0
        recurrent_obs = recurrent_obs_val * np.ones((1, OBS_SIZE), dtype=np.float32)

        m_vector_obs = [np.concatenate([base_obs, recurrent_obs], axis=1)]
        m_reward = np.array([reward], dtype=np.float32)
        m_done = np.array([done], dtype=np.bool)
        m_agent_id = np.array([0], dtype=np.int32)
        action_mask = self._generate_mask()
        return BatchedStepResult(
            m_vector_obs, m_reward, m_done, m_done, m_agent_id, action_mask
        )

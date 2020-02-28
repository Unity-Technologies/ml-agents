import random
from typing import Dict
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

    def __init__(self, brain_names, use_discrete):
        super().__init__()
        self.discrete = use_discrete
        action_type = ActionType.DISCRETE if use_discrete else ActionType.CONTINUOUS
        self.group_spec = AgentGroupSpec(
            [(OBS_SIZE,)], action_type, (2,) if use_discrete else 1
        )
        # state
        self.names = brain_names
        self.position: Dict[str, float] = {}
        self.step_count: Dict[str, float] = {}
        self.random = random.Random(str(self.group_spec))
        self.goal = self.random.choice([-1, 1])
        self.action = {}
        self.rewards: Dict[str, float] = {}
        self.final_rewards: Dict[str, float] = {}
        self.step_result: Dict[str, BatchedStepResult] = {}

        for name in self.names:
            self.rewards[name] = 0
            self._reset_agent(name)
            self.final_rewards[name] = 0
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

    def step(self) -> None:
        assert all(action is not None for action in self.action.values())

        for name in self.names:
            if self.discrete:
                act = self.action[name][0][0]
                delta = 1 if act else -1
            else:
                delta = self.action[name][0][0]
            delta = clamp(delta, -STEP_SIZE, STEP_SIZE)
            self.position[name] += delta
            self.position[name] = clamp(self.position[name], -1, 1)
            self.step_count[name] += 1
            done = self.position[name] >= 1.0 or self.position[name] <= -1.0
            if done:
                reward = SUCCESS_REWARD * self.position[name] * self.goal
            else:
                reward = -TIME_PENALTY
            self.rewards[name] += reward

            m_vector_obs = [np.zeros((1, OBS_SIZE), dtype=np.float32)]
            m_reward = np.array([reward], dtype=np.float32)
            m_done = np.array([done], dtype=np.bool)
            m_agent_id = np.array([0], dtype=np.int32)
            action_mask = self._generate_mask()

            if done:
                self._reset_agent(name)

            self.step_result[name] = BatchedStepResult(
                m_vector_obs, m_reward, m_done, m_done, m_agent_id, action_mask
            )

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
        self.position[name] = 0.0
        self.step_count[name] = 0
        self.final_rewards[name] = self.rewards[name]
        self.rewards[name] = 0

    def reset(self) -> None:  # type: ignore
        for name in self.names:
            self._reset_agent(name)

            m_vector_obs = [np.zeros((1, OBS_SIZE), dtype=np.float32)]
            m_reward = np.array([0], dtype=np.float32)
            m_done = np.array([False], dtype=np.bool)
            m_agent_id = np.array([0], dtype=np.int32)
            action_mask = self._generate_mask()

            self.step_result[name] = BatchedStepResult(
                m_vector_obs, m_reward, m_done, m_done, m_agent_id, action_mask
            )
        # self.goal = self.random.choice([-1, 1])

    @property
    def reset_parameters(self) -> Dict[str, str]:
        return {}

    def close(self):
        pass


# class Ghost1DEnv(Simple1DEnvironment):
#    """
#    Extends the 1D environment to include another agent.  This is not
#    adversarial but checks to see that the ghost trainer (1) trains an agent successfully
#    and (2) properly ghosts another agent. The two agents are distinguished by team id.
#    """
#
#    def __init__(self, use_discrete):
#        super().__init__(use_discrete)
#
# def get_agent_groups(self):
#        return [BRAIN_NAME+"?team=0", BRAIN_NAME+"?team=1"]
#

import random
from typing import Dict, List, Any, Tuple
import numpy as np

from mlagents_envs.base_env import (
    BaseEnv,
    AgentGroupSpec,
    BatchedStepResult,
    ActionType,
)
from mlagents_envs.tests.test_rpc_utils import proto_from_batched_step_result_and_action
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)

OBS_SIZE = 1
VIS_OBS_SIZE = (20, 20, 3)
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

    def __init__(
        self,
        brain_names,
        use_discrete,
        step_size=STEP_SIZE,
        num_visual=0,
        num_vector=1,
        vis_obs_size=VIS_OBS_SIZE,
        vec_obs_size=OBS_SIZE,
    ):
        super().__init__()
        self.discrete = use_discrete
        self.num_visual = num_visual
        self.num_vector = num_vector
        self.vis_obs_size = vis_obs_size
        self.vec_obs_size = vec_obs_size
        action_type = ActionType.DISCRETE if use_discrete else ActionType.CONTINUOUS
        self.group_spec = AgentGroupSpec(
            self._make_obs_spec(), action_type, (2,) if use_discrete else 1
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
        self.agent_id: Dict[str, int] = {}
        self.step_size = step_size  # defines the difficulty of the test

        for name in self.names:
            self.agent_id[name] = 0
            self.goal[name] = self.random.choice([-1, 1])
            self.rewards[name] = 0
            self.final_rewards[name] = []
            self._reset_agent(name)
            self.action[name] = None
            self.step_result[name] = None

    def _make_obs_spec(self) -> List[Any]:
        obs_spec: List[Any] = []
        for _ in range(self.num_vector):
            obs_spec.append((self.vec_obs_size,))
        for _ in range(self.num_visual):
            obs_spec.append(self.vis_obs_size)
        return obs_spec

    def _make_obs(self, value: float) -> List[np.ndarray]:
        obs = []
        for _ in range(self.num_vector):
            obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * value)
        for _ in range(self.num_visual):
            obs.append(np.ones((1,) + self.vis_obs_size, dtype=np.float32) * value)
        return obs

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
        m_vector_obs = self._make_obs(self.goal[name])
        m_reward = np.array([reward], dtype=np.float32)
        m_done = np.array([done], dtype=np.bool)
        m_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        action_mask = self._generate_mask()

        if done:
            self._reset_agent(name)
            new_vector_obs = self._make_obs(self.goal[name])
            (
                m_vector_obs,
                m_reward,
                m_done,
                m_agent_id,
                action_mask,
            ) = self._construct_reset_step(
                m_vector_obs,
                new_vector_obs,
                m_reward,
                m_done,
                m_agent_id,
                action_mask,
                name,
            )
        return BatchedStepResult(
            m_vector_obs,
            m_reward,
            m_done,
            np.zeros(m_done.shape, dtype=bool),
            m_agent_id,
            action_mask,
        )

    def _construct_reset_step(
        self,
        vector_obs: List[np.ndarray],
        new_vector_obs: List[np.ndarray],
        reward: np.ndarray,
        done: np.ndarray,
        agent_id: np.ndarray,
        action_mask: List[np.ndarray],
        name: str,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        new_reward = np.array([0.0], dtype=np.float32)
        new_done = np.array([False], dtype=np.bool)
        new_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        new_action_mask = self._generate_mask()

        m_vector_obs = [
            np.concatenate((old, new), axis=0)
            for old, new in zip(vector_obs, new_vector_obs)
        ]
        m_reward = np.concatenate((reward, new_reward), axis=0)
        m_done = np.concatenate((done, new_done), axis=0)
        m_agent_id = np.concatenate((agent_id, new_agent_id), axis=0)
        if action_mask is not None:
            action_mask = [
                np.concatenate((old, new), axis=0)
                for old, new in zip(action_mask, new_action_mask)
            ]
        return m_vector_obs, m_reward, m_done, m_agent_id, action_mask

    def step(self) -> None:
        assert all(action is not None for action in self.action.values())
        for name in self.names:

            done = self._take_action(name)
            reward = self._compute_reward(name, done)
            self.rewards[name] += reward
            self.step_result[name] = self._make_batched_step(name, done, reward)

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
        self.agent_id[name] = self.agent_id[name] + 1

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
    def __init__(self, brain_names, use_discrete, step_size=0.2):
        super().__init__(brain_names, use_discrete, step_size=step_size)
        # Number of steps to reveal the goal for. Lower is harder. Should be
        # less than 1/step_size to force agent to use memory
        self.num_show_steps = 2

    def _make_batched_step(
        self, name: str, done: bool, reward: float
    ) -> BatchedStepResult:
        recurrent_obs_val = (
            self.goal[name] if self.step_count[name] <= self.num_show_steps else 0
        )
        m_vector_obs = self._make_obs(recurrent_obs_val)
        m_reward = np.array([reward], dtype=np.float32)
        m_done = np.array([done], dtype=np.bool)
        m_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        action_mask = self._generate_mask()
        if done:
            self._reset_agent(name)
            recurrent_obs_val = (
                self.goal[name] if self.step_count[name] <= self.num_show_steps else 0
            )
            new_vector_obs = self._make_obs(recurrent_obs_val)
            (
                m_vector_obs,
                m_reward,
                m_done,
                m_agent_id,
                action_mask,
            ) = self._construct_reset_step(
                m_vector_obs,
                new_vector_obs,
                m_reward,
                m_done,
                m_agent_id,
                action_mask,
                name,
            )
        return BatchedStepResult(
            m_vector_obs,
            m_reward,
            m_done,
            np.zeros(m_done.shape, dtype=bool),
            m_agent_id,
            action_mask,
        )


class Record1DEnvironment(Simple1DEnvironment):
    def __init__(
        self,
        brain_names,
        use_discrete,
        step_size=0.2,
        num_visual=0,
        num_vector=1,
        n_demos=30,
    ):
        super().__init__(
            brain_names,
            use_discrete,
            step_size=step_size,
            num_visual=num_visual,
            num_vector=num_vector,
        )
        self.demonstration_protos: Dict[str, List[AgentInfoActionPairProto]] = {}
        self.n_demos = n_demos
        for name in self.names:
            self.demonstration_protos[name] = []

    def step(self) -> None:
        super().step()
        for name in self.names:
            self.demonstration_protos[
                name
            ] += proto_from_batched_step_result_and_action(
                self.step_result[name], self.action[name]
            )
            self.demonstration_protos[name] = self.demonstration_protos[name][
                -self.n_demos :
            ]

    def solve(self) -> None:
        self.reset()
        for _ in range(self.n_demos):
            for name in self.names:
                if self.discrete:
                    self.action[name] = [[1]] if self.goal[name] > 0 else [[0]]
                else:
                    self.action[name] = [[float(self.goal[name])]]
            self.step()

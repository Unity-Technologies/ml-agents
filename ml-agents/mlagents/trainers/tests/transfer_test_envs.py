import random
from typing import Dict, List, Any, Tuple
import numpy as np

from mlagents_envs.base_env import (
    BaseEnv,
    BehaviorSpec,
    DecisionSteps,
    TerminalSteps,
    ActionType,
    BehaviorMapping,
)
from mlagents_envs.tests.test_rpc_utils import proto_from_steps_and_action
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)

OBS_SIZE = 1
VIS_OBS_SIZE = (20, 20, 3)
STEP_SIZE = 0.1

TIME_PENALTY = 0.01
MIN_STEPS = int(1.0 / STEP_SIZE) + 1
SUCCESS_REWARD = 1.0 + MIN_STEPS * TIME_PENALTY


def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))


class SimpleTransferEnvironment(BaseEnv):
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
        action_size=1,
        obs_spec_type="normal", # normal: (x,y); rich: (x+y, x-y, x*y)
        goal_type="hard", # easy: 1 or -1; hard: uniformly random
        act_speed=1
    ):
        super().__init__()
        self.discrete = use_discrete
        self.num_visual = num_visual
        self.num_vector = num_vector
        self.vis_obs_size = vis_obs_size
        self.vec_obs_size = vec_obs_size
        self.obs_spec_type = obs_spec_type
        self.goal_type = goal_type
        action_type = ActionType.DISCRETE if use_discrete else ActionType.CONTINUOUS
        self.behavior_spec = BehaviorSpec(
            self._make_obs_spec(),
            action_type,
            tuple(2 for _ in range(action_size)) if use_discrete else action_size,
        )
        self.action_size = action_size
        self.names = brain_names
        self.positions: Dict[str, List[float]] = {}
        self.step_count: Dict[str, float] = {}
        self.random = random.Random(str(self.behavior_spec))
        self.goal: Dict[str, List[float]] = {}
        self.num_steps: Dict[str, int] = {}
        self.horizon: Dict[str, int] = {}
        self.action = {}
        self.rewards: Dict[str, float] = {}
        self.final_rewards: Dict[str, List[float]] = {}
        self.step_result: Dict[str, Tuple[DecisionSteps, TerminalSteps]] = {}
        self.agent_id: Dict[str, int] = {}
        self.step_size = step_size  # defines the difficulty of the test
        self.act_speed = act_speed

        for name in self.names:
            self.agent_id[name] = 0
            if self.goal_type == "easy":
                self.goal[name] = []
                for _ in range(self.num_vector):
                    self.goal[name].append(self.random.choice([-1, 1]))
            elif self.goal_type == "hard":
                self.goal[name] = []
                for _ in range(self.num_vector):
                    self.goal[name].append(self.random.uniform(-1,1))
            self.rewards[name] = 0
            self.final_rewards[name] = []
            self._reset_agent(name)
            self.action[name] = None
            self.step_result[name] = None
            self.step_count[name] = 0
            self.horizon[name] = 1000
        print(self.goal)

    def _make_obs_spec(self) -> List[Any]:
        obs_spec: List[Any] = []
        # goal
        for _ in range(self.num_vector):
            obs_spec.append((self.vec_obs_size,))
        for _ in range(self.num_visual):
            obs_spec.append(self.vis_obs_size)
        # position
        if self.obs_spec_type == "normal":
            for _ in range(self.num_vector):
                obs_spec.append((self.vec_obs_size,))
        # composed position
        elif "rich" in self.obs_spec_type:
            for _ in range(self.num_vector+1):
                obs_spec.append((self.vec_obs_size,))
        print("obs_spec:", obs_spec)
        return obs_spec

    def _make_obs(self, value: List[float]) -> List[np.ndarray]:
        obs = []
        if self.obs_spec_type == "compact":
            for name in self.names:
                for pos, goal in zip(self.positions[name], value):
                    obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (goal-pos))
            return obs

        for i in range(self.num_vector):
            obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * value[i])
            
        if self.obs_spec_type == "normal":
            for name in self.names:
                for i in self.positions[name]:
                    obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * i)
        elif self.obs_spec_type == "rich1":
            for name in self.names:
                i = self.positions[name][0]
                j = self.positions[name][1]
                obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (i+j))
                obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (i-j))
                obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (i*j))
        elif self.obs_spec_type == "rich2":
            for name in self.names:
                i = self.positions[name][0]
                j = self.positions[name][1]
                obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (i*j))
                obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (2*i+j))
                obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * (2*i-j))
        for _ in range(self.num_visual):
            obs.append(np.ones((1,) + self.vis_obs_size, dtype=np.float32) * value)
        return obs

    @property
    def behavior_specs(self):
        behavior_dict = {}
        for n in self.names:
            behavior_dict[n] = self.behavior_spec
        return BehaviorMapping(behavior_dict)

    def set_action_for_agent(self, behavior_name, agent_id, action):
        pass

    def set_actions(self, behavior_name, action):
        self.action[behavior_name] = action

    def get_steps(self, behavior_name):
        return self.step_result[behavior_name]

    def _take_action(self, name: str) -> bool:
        deltas = []
        for _act in self.action[name][0]:
            if self.discrete:
                deltas.append(1 if _act else -1)
            else:
                deltas.append(_act)
        for i, _delta in enumerate(deltas):
            _delta = clamp(_delta, -self.step_size, self.step_size)
            self.positions[name][i] += _delta
            self.positions[name][i] = clamp(self.positions[name][i], -1, 1)
            self.step_count[name] += 1
            # Both must be in 1.0 to be done
        # print(self.positions[name], end="")
        if self.goal_type == "easy":
            done = all(pos >= 1.0 or pos <= -1.0 for pos in self.positions[name]) or self.step_count[name] >= self.horizon[name]
        elif self.goal_type == "hard":
            # done = self.step_count[name] >= self.horizon[name]
            done = all(abs(pos-goal) <= 0.1 for pos, goal in zip(self.positions[name], self.goal[name])) \
                 or self.step_count[name] >= self.horizon[name]
        # if done:
        #     print(self.positions[name], end=" done ")
        return done

    def _generate_mask(self):
        if self.discrete:
            # LL-Python API will return an empty dim if there is only 1 agent.
            ndmask = np.array(2 * self.action_size * [False], dtype=np.bool)
            ndmask = np.expand_dims(ndmask, axis=0)
            action_mask = [ndmask]
        else:
            action_mask = None
        return action_mask

    def _compute_reward(self, name: str, done: bool) -> float:
        # reward = 0.0
        # for _pos, goal in zip(self.positions[name], self.goal[name]):
        # #     if abs(_pos - self.goal[name]) < 0.1:
        # #         reward += SUCCESS_REWARD
        # #     else:
        # #         reward -= TIME_PENALTY
        #     reward -= abs(_pos - goal) / 10 #np.exp(-abs(_pos - goal))

        if done and self.step_count[name] < self.horizon[name]:
            reward = 0.0 #SUCCESS_REWARD
            # for _pos in self.positions[name]:
            #     if self.goal_type == "easy":
            #         reward += (SUCCESS_REWARD * _pos * self.goal[name]) / len(
            #             self.positions[name]
            #         )
            #     elif self.goal_type == "hard":
            #         reward += np.exp(-abs(_pos - self.goal[name]))
        else:
            reward = -TIME_PENALTY

        return reward

    def _reset_agent(self, name):
        if self.goal_type == "easy":
            self.goal[name] = []
            for _ in range(self.num_vector):
                self.goal[name].append(self.random.choice([-1, 1]))
        elif self.goal_type == "hard":
            self.goal[name] = []
            for _ in range(self.num_vector):
                self.goal[name].append(self.random.uniform(-1,1))
        self.positions[name] = [self.random.uniform(-1,1) for _ in range(self.action_size)]
        self.step_count[name] = 0
        self.rewards[name] = 0
        self.agent_id[name] = self.agent_id[name] + 1
        # print("new goal:", self.goal[name])
        # print("new pos:", self.positions[name])

    def _make_batched_step(
        self, name: str, done: bool, reward: float
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        m_vector_obs = self._make_obs(self.goal[name])
        m_reward = np.array([reward], dtype=np.float32)
        m_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        action_mask = self._generate_mask()
        decision_step = DecisionSteps(m_vector_obs, m_reward, m_agent_id, action_mask)
        terminal_step = TerminalSteps.empty(self.behavior_spec)
        if done:
            self.final_rewards[name].append(self.rewards[name])
            self._reset_agent(name)
            new_vector_obs = self._make_obs(self.goal[name])
            (
                new_reward,
                new_done,
                new_agent_id,
                new_action_mask,
            ) = self._construct_reset_step(name)

            decision_step = DecisionSteps(
                new_vector_obs, new_reward, new_agent_id, new_action_mask
            )
            terminal_step = TerminalSteps(
                m_vector_obs, m_reward, np.array([False], dtype=np.bool), m_agent_id
            )
        return (decision_step, terminal_step)

    def _construct_reset_step(
        self, name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        new_reward = np.array([0.0], dtype=np.float32)
        new_done = np.array([False], dtype=np.bool)
        new_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        new_action_mask = self._generate_mask()
        return new_reward, new_done, new_agent_id, new_action_mask

    def step(self) -> None:
        assert all(action is not None for action in self.action.values())
        for name in self.names:

            done = self._take_action(name)
            reward = self._compute_reward(name, done)
            self.rewards[name] += reward
            self.step_result[name] = self._make_batched_step(name, done, reward)

    def reset(self) -> None:  # type: ignore
        for name in self.names:
            self._reset_agent(name)
            self.step_result[name] = self._make_batched_step(name, False, 0.0)

    @property
    def reset_parameters(self) -> Dict[str, str]:
        return {}

    def close(self):
        pass

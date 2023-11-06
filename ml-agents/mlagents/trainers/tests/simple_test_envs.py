import random
from typing import Dict, List, Any, Tuple
import numpy as np

from mlagents_envs.base_env import (
    ActionSpec,
    ObservationSpec,
    ObservationType,
    ActionTuple,
    BaseEnv,
    BehaviorSpec,
    DecisionSteps,
    TerminalSteps,
    BehaviorMapping,
)
from .test_rpc_utils import proto_from_steps_and_action
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import (
    AgentInfoActionPairProto,
)
from mlagents.trainers.tests.dummy_config import create_observation_specs_with_shapes

OBS_SIZE = 1
VIS_OBS_SIZE = (3, 20, 20)
VAR_LEN_SIZE = (10, 5)
STEP_SIZE = 0.2

TIME_PENALTY = 0.01
MIN_STEPS = int(1.0 / STEP_SIZE) + 1
SUCCESS_REWARD = 1.0 + MIN_STEPS * TIME_PENALTY


def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))


class SimpleEnvironment(BaseEnv):
    """
    Very simple "game" - the agent has a position on [-1, 1], gets a reward of 1 if it reaches 1, and a reward of -1 if
    it reaches -1. The position is incremented by the action amount (clamped to [-step_size, step_size]).
    """

    def __init__(
        self,
        brain_names,
        step_size=STEP_SIZE,
        num_visual=0,
        num_vector=1,
        num_var_len=0,
        vis_obs_size=VIS_OBS_SIZE,
        vec_obs_size=OBS_SIZE,
        var_len_obs_size=VAR_LEN_SIZE,
        action_sizes=(1, 0),
        goal_indices=None,
    ):
        super().__init__()
        self.num_visual = num_visual
        self.num_vector = num_vector
        self.num_var_len = num_var_len
        self.vis_obs_size = vis_obs_size
        self.vec_obs_size = vec_obs_size
        self.var_len_obs_size = var_len_obs_size
        self.goal_indices = goal_indices
        continuous_action_size, discrete_action_size = action_sizes
        discrete_tuple = tuple(2 for _ in range(discrete_action_size))
        action_spec = ActionSpec(continuous_action_size, discrete_tuple)
        self.total_action_size = (
            continuous_action_size + discrete_action_size
        )  # to set the goals/positions
        self.action_spec = action_spec
        self.behavior_spec = BehaviorSpec(self._make_observation_specs(), action_spec)
        self.action_spec = action_spec
        self.names = brain_names
        self.positions: Dict[str, List[float]] = {}
        self.step_count: Dict[str, float] = {}

        # Concatenate the arguments for a consistent random seed
        seed = (
            brain_names,
            step_size,
            num_visual,
            num_vector,
            num_var_len,
            vis_obs_size,
            vec_obs_size,
            var_len_obs_size,
            action_sizes,
        )
        self.random = random.Random(str(seed))

        self.goal: Dict[str, int] = {}
        self.action = {}
        self.rewards: Dict[str, float] = {}
        self.final_rewards: Dict[str, List[float]] = {}
        self.step_result: Dict[str, Tuple[DecisionSteps, TerminalSteps]] = {}
        self.agent_id: Dict[str, int] = {}
        self.step_size = step_size  # defines the difficulty of the test
        # Allow to be used as a UnityEnvironment during tests
        self.academy_capabilities = None

        for name in self.names:
            self.agent_id[name] = 0
            self.goal[name] = self.random.choice([-1, 1])
            self.rewards[name] = 0
            self.final_rewards[name] = []
            self._reset_agent(name)
            self.action[name] = None
            self.step_result[name] = None

    def _make_observation_specs(self) -> List[ObservationSpec]:
        obs_shape: List[Any] = []
        for _ in range(self.num_vector):
            obs_shape.append((self.vec_obs_size,))
        for _ in range(self.num_visual):
            obs_shape.append(self.vis_obs_size)
        for _ in range(self.num_var_len):
            obs_shape.append(self.var_len_obs_size)
        obs_spec = create_observation_specs_with_shapes(obs_shape)
        if self.goal_indices is not None:
            for i in range(len(obs_spec)):
                if i in self.goal_indices:
                    obs_spec[i] = ObservationSpec(
                        shape=obs_spec[i].shape,
                        dimension_property=obs_spec[i].dimension_property,
                        observation_type=ObservationType.GOAL_SIGNAL,
                        name=obs_spec[i].name,
                    )
        return obs_spec

    def _make_obs(self, value: float) -> List[np.ndarray]:
        obs = []
        for _ in range(self.num_vector):
            obs.append(np.ones((1, self.vec_obs_size), dtype=np.float32) * value)
        for _ in range(self.num_visual):
            obs.append(np.ones((1,) + self.vis_obs_size, dtype=np.float32) * value)
        for _ in range(self.num_var_len):
            obs.append(np.ones((1,) + self.var_len_obs_size, dtype=np.float32) * value)
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
        _act = self.action[name]
        if self.action_spec.continuous_size > 0:
            for _cont in _act.continuous[0]:
                deltas.append(_cont)
        if self.action_spec.discrete_size > 0:
            for _disc in _act.discrete[0]:
                deltas.append(1 if _disc else -1)
        for i, _delta in enumerate(deltas):
            _delta = clamp(_delta, -self.step_size, self.step_size)
            self.positions[name][i] += _delta
            self.positions[name][i] = clamp(self.positions[name][i], -1, 1)
            self.step_count[name] += 1
            # Both must be in 1.0 to be done
        done = all(pos >= 1.0 or pos <= -1.0 for pos in self.positions[name])
        return done

    def _generate_mask(self):
        action_mask = None
        if self.action_spec.discrete_size > 0:
            # LL-Python API will return an empty dim if there is only 1 agent.
            ndmask = np.array(
                2 * self.action_spec.discrete_size * [False], dtype=np.bool
            )
            ndmask = np.expand_dims(ndmask, axis=0)
            action_mask = [ndmask]
        return action_mask

    def _compute_reward(self, name: str, done: bool) -> float:
        if done:
            reward = 0.0
            for _pos in self.positions[name]:
                reward += (SUCCESS_REWARD * _pos * self.goal[name]) / len(
                    self.positions[name]
                )
        else:
            reward = -TIME_PENALTY
        return reward

    def _reset_agent(self, name):
        self.goal[name] = self.random.choice([-1, 1])
        self.positions[name] = [0.0 for _ in range(self.total_action_size)]
        self.step_count[name] = 0
        self.rewards[name] = 0
        self.agent_id[name] = self.agent_id[name] + 1

    def _make_batched_step(
        self, name: str, done: bool, reward: float, group_reward: float
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        m_vector_obs = self._make_obs(self.goal[name])
        m_reward = np.array([reward], dtype=np.float32)
        m_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        m_group_id = np.array([0], dtype=np.int32)
        m_group_reward = np.array([group_reward], dtype=np.float32)
        action_mask = self._generate_mask()
        decision_step = DecisionSteps(
            m_vector_obs, m_reward, m_agent_id, action_mask, m_group_id, m_group_reward
        )
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
                new_group_id,
                new_group_reward,
            ) = self._construct_reset_step(name)

            decision_step = DecisionSteps(
                new_vector_obs,
                new_reward,
                new_agent_id,
                new_action_mask,
                new_group_id,
                new_group_reward,
            )
            terminal_step = TerminalSteps(
                m_vector_obs,
                m_reward,
                np.array([False], dtype=np.bool),
                m_agent_id,
                m_group_id,
                m_group_reward,
            )
        return (decision_step, terminal_step)

    def _construct_reset_step(
        self, name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        new_reward = np.array([0.0], dtype=np.float32)
        new_done = np.array([False], dtype=np.bool)
        new_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        new_action_mask = self._generate_mask()
        new_group_id = np.array([0], dtype=np.int32)
        new_group_reward = np.array([0.0], dtype=np.float32)
        return (
            new_reward,
            new_done,
            new_agent_id,
            new_action_mask,
            new_group_id,
            new_group_reward,
        )

    def step(self) -> None:
        assert all(action is not None for action in self.action.values())
        for name in self.names:

            done = self._take_action(name)
            reward = self._compute_reward(name, done)
            self.rewards[name] += reward
            self.step_result[name] = self._make_batched_step(name, done, reward, 0.0)

    def reset(self) -> None:  # type: ignore
        for name in self.names:
            self._reset_agent(name)
            self.step_result[name] = self._make_batched_step(name, False, 0.0, 0.0)

    @property
    def reset_parameters(self) -> Dict[str, str]:
        return {}

    def close(self):
        pass


class MemoryEnvironment(SimpleEnvironment):
    def __init__(self, brain_names, action_sizes=(1, 0), step_size=0.2):
        super().__init__(brain_names, action_sizes=action_sizes, step_size=step_size)
        # Number of steps to reveal the goal for. Lower is harder. Should be
        # less than 1/step_size to force agent to use memory
        self.num_show_steps = 2

    def _make_batched_step(
        self, name: str, done: bool, reward: float, group_reward: float
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        recurrent_obs_val = (
            self.goal[name] if self.step_count[name] <= self.num_show_steps else 0
        )
        m_vector_obs = self._make_obs(recurrent_obs_val)
        m_reward = np.array([reward], dtype=np.float32)
        m_agent_id = np.array([self.agent_id[name]], dtype=np.int32)
        m_group_id = np.array([0], dtype=np.int32)
        m_group_reward = np.array([group_reward], dtype=np.float32)
        action_mask = self._generate_mask()
        decision_step = DecisionSteps(
            m_vector_obs, m_reward, m_agent_id, action_mask, m_group_id, m_group_reward
        )
        terminal_step = TerminalSteps.empty(self.behavior_spec)
        if done:
            self.final_rewards[name].append(self.rewards[name])
            self._reset_agent(name)
            recurrent_obs_val = (
                self.goal[name] if self.step_count[name] <= self.num_show_steps else 0
            )
            new_vector_obs = self._make_obs(recurrent_obs_val)
            (
                new_reward,
                new_done,
                new_agent_id,
                new_action_mask,
                new_group_id,
                new_group_reward,
            ) = self._construct_reset_step(name)
            decision_step = DecisionSteps(
                new_vector_obs,
                new_reward,
                new_agent_id,
                new_action_mask,
                new_group_id,
                new_group_reward,
            )
            terminal_step = TerminalSteps(
                m_vector_obs,
                m_reward,
                np.array([False], dtype=np.bool),
                m_agent_id,
                m_group_id,
                m_group_reward,
            )
        return (decision_step, terminal_step)


class MultiAgentEnvironment(BaseEnv):
    """
    The MultiAgentEnvironment maintains a list of SimpleEnvironment, one for each agent.
    When sending DecisionSteps and TerminalSteps to the trainers, it first batches the
    decision steps from the individual environments. When setting actions, it indexes the
    batched ActionTuple to obtain the ActionTuple for individual agents
    """

    def __init__(
        self,
        brain_names,
        step_size=STEP_SIZE,
        num_visual=0,
        num_vector=1,
        num_var_len=0,
        vis_obs_size=VIS_OBS_SIZE,
        vec_obs_size=OBS_SIZE,
        var_len_obs_size=VAR_LEN_SIZE,
        action_sizes=(1, 0),
        num_agents=2,
        goal_indices=None,
    ):
        super().__init__()
        self.envs = {}
        self.dones = {}
        self.just_died = set()
        self.names = brain_names
        self.final_rewards: Dict[str, List[float]] = {}
        for name in brain_names:
            self.final_rewards[name] = []
            for i in range(num_agents):
                name_and_num = name + str(i)
                self.envs[name_and_num] = SimpleEnvironment(
                    [name],
                    step_size,
                    num_visual,
                    num_vector,
                    num_var_len,
                    vis_obs_size,
                    vec_obs_size,
                    var_len_obs_size,
                    action_sizes,
                    goal_indices,
                )
                self.dones[name_and_num] = False
                self.envs[name_and_num].reset()
        # All envs have the same behavior spec, so just get the last one.
        self.behavior_spec = self.envs[name_and_num].behavior_spec
        self.action_spec = self.envs[name_and_num].action_spec
        self.num_agents = num_agents

    @property
    def all_done(self):
        return all(self.dones.values())

    @property
    def behavior_specs(self):
        behavior_dict = {}
        for n in self.names:
            behavior_dict[n] = self.behavior_spec
        return BehaviorMapping(behavior_dict)

    def set_action_for_agent(self, behavior_name, agent_id, action):
        pass

    def set_actions(self, behavior_name, action):
        # The ActionTuple contains the actions for all n_agents. This
        # slices the ActionTuple into an action tuple for each environment
        # and sets it. The index j is used to ignore agents that have already
        # reached done.
        j = 0
        for i in range(self.num_agents):
            _act = ActionTuple()
            name_and_num = behavior_name + str(i)
            env = self.envs[name_and_num]
            if not self.dones[name_and_num]:
                if self.action_spec.continuous_size > 0:
                    _act.add_continuous(action.continuous[j : j + 1])
                if self.action_spec.discrete_size > 0:
                    _disc_list = [action.discrete[j, :]]
                    _act.add_discrete(np.array(_disc_list))
                j += 1
                env.action[behavior_name] = _act

    def get_steps(self, behavior_name):
        # This gets the individual DecisionSteps and TerminalSteps
        # from the envs and merges them into a batch to be sent
        # to the AgentProcessor.
        dec_vec_obs = []
        dec_reward = []
        dec_group_reward = []
        dec_agent_id = []
        dec_group_id = []
        ter_vec_obs = []
        ter_reward = []
        ter_group_reward = []
        ter_agent_id = []
        ter_group_id = []
        interrupted = []

        action_mask = None
        terminal_step = TerminalSteps.empty(self.behavior_spec)
        decision_step = None
        for i in range(self.num_agents):
            name_and_num = behavior_name + str(i)
            env = self.envs[name_and_num]
            _dec, _term = env.step_result[behavior_name]
            if not self.dones[name_and_num]:
                dec_agent_id.append(i)
                dec_group_id.append(1)
                if len(dec_vec_obs) > 0:
                    for j, obs in enumerate(_dec.obs):
                        dec_vec_obs[j] = np.concatenate((dec_vec_obs[j], obs), axis=0)
                else:
                    for obs in _dec.obs:
                        dec_vec_obs.append(obs)
                dec_reward.append(_dec.reward[0])
                dec_group_reward.append(_dec.group_reward[0])
                if _dec.action_mask is not None:
                    if action_mask is None:
                        action_mask = []
                    if len(action_mask) > 0:
                        action_mask[0] = np.concatenate(
                            (action_mask[0], _dec.action_mask[0]), axis=0
                        )
                    else:
                        action_mask.append(_dec.action_mask[0])
            if len(_term.reward) > 0 and name_and_num in self.just_died:
                ter_agent_id.append(i)
                ter_group_id.append(1)
                if len(ter_vec_obs) > 0:
                    for j, obs in enumerate(_term.obs):
                        ter_vec_obs[j] = np.concatenate((ter_vec_obs[j], obs), axis=0)
                else:
                    for obs in _term.obs:
                        ter_vec_obs.append(obs)
                ter_reward.append(_term.reward[0])
                ter_group_reward.append(_term.group_reward[0])
                interrupted.append(False)
                self.just_died.remove(name_and_num)
        decision_step = DecisionSteps(
            dec_vec_obs,
            dec_reward,
            dec_agent_id,
            action_mask,
            dec_group_id,
            dec_group_reward,
        )
        terminal_step = TerminalSteps(
            ter_vec_obs,
            ter_reward,
            interrupted,
            ter_agent_id,
            ter_group_id,
            ter_group_reward,
        )
        return (decision_step, terminal_step)

    def step(self) -> None:
        # Steps all environments and calls reset if all agents are done.
        for name in self.names:
            for i in range(self.num_agents):
                name_and_num = name + str(i)
                # Does not step the env if done
                if not self.dones[name_and_num]:
                    env = self.envs[name_and_num]
                    # Reproducing part of env step to intercept Dones
                    assert all(action is not None for action in env.action.values())
                    done = env._take_action(name)
                    reward = env._compute_reward(name, done)
                    self.dones[name_and_num] = done
                    if done:
                        self.just_died.add(name_and_num)
                    if self.all_done:
                        env.step_result[name] = env._make_batched_step(
                            name, done, 0.0, reward
                        )
                        self.final_rewards[name].append(reward)
                        self.reset()
                    elif done:
                        # This agent has finished but others are still running.
                        # This gives a reward of the time penalty if this agent
                        # is successful and the negative env reward if it fails.
                        ceil_reward = min(-TIME_PENALTY, reward)
                        env.step_result[name] = env._make_batched_step(
                            name, done, ceil_reward, 0.0
                        )
                        self.final_rewards[name].append(reward)

                    else:
                        env.step_result[name] = env._make_batched_step(
                            name, done, reward, 0.0
                        )

    def reset(self) -> None:  # type: ignore
        for name in self.names:
            for i in range(self.num_agents):
                name_and_num = name + str(i)
                self.dones[name_and_num] = False

    @property
    def reset_parameters(self) -> Dict[str, str]:
        return {}

    def close(self):
        pass


class RecordEnvironment(SimpleEnvironment):
    def __init__(
        self,
        brain_names,
        step_size=0.2,
        num_visual=0,
        num_vector=1,
        action_sizes=(1, 0),
        n_demos=30,
    ):
        super().__init__(
            brain_names,
            step_size=step_size,
            num_visual=num_visual,
            num_vector=num_vector,
            action_sizes=action_sizes,
        )
        self.demonstration_protos: Dict[str, List[AgentInfoActionPairProto]] = {}
        self.n_demos = n_demos
        for name in self.names:
            self.demonstration_protos[name] = []

    def step(self) -> None:
        super().step()
        for name in self.names:
            discrete_actions = (
                self.action[name].discrete
                if self.action_spec.discrete_size > 0
                else None
            )
            continuous_actions = (
                self.action[name].continuous
                if self.action_spec.continuous_size > 0
                else None
            )
            self.demonstration_protos[name] += proto_from_steps_and_action(
                self.step_result[name][0],
                self.step_result[name][1],
                continuous_actions,
                discrete_actions,
            )
            self.demonstration_protos[name] = self.demonstration_protos[name][
                -self.n_demos :
            ]

    def solve(self) -> None:
        self.reset()
        for _ in range(self.n_demos):
            for name in self.names:
                if self.action_spec.discrete_size > 0:
                    self.action[name] = ActionTuple(
                        np.array([], dtype=np.float32),
                        np.array(
                            [[1]] if self.goal[name] > 0 else [[0]], dtype=np.int32
                        ),
                    )
                else:
                    self.action[name] = ActionTuple(
                        np.array([[float(self.goal[name])]], dtype=np.float32),
                        np.array([], dtype=np.int32),
                    )
            self.step()


class UnexpectedExceptionEnvironment(SimpleEnvironment):
    def __init__(self, brain_names, use_discrete, to_raise):
        super().__init__(brain_names, use_discrete)
        self.to_raise = to_raise

    def step(self) -> None:
        raise self.to_raise()

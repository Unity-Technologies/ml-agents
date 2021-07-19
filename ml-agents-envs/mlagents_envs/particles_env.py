import numpy as np
from typing import List, Tuple, Optional, Mapping as MappingType
from mlagents_envs.base_env import (
    BehaviorSpec,
    ObservationSpec,
    DimensionProperty,
    ObservationType,
    ActionSpec,
    DecisionSteps,
    TerminalSteps,
    BaseEnv,
    BehaviorName,
    ActionTuple,
    AgentId,
)
from mlagents_envs.communicator_objects.capabilities_pb2 import UnityRLCapabilitiesProto


def _make_env(scenario_name, benchmark=False):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from multiagent.environment import MultiAgentEnv
    from multiagent import scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


class ParticlesEnvironment(BaseEnv):
    def __init__(self, name: str = "simple_spread"):
        self._obs: Optional[List[np.array]] = None
        self._rew: Optional[List[int]] = None
        self._done: Optional[List[bool]] = None
        self._actions: Optional[List[int]] = None
        self._name = name
        self._env = _make_env(name)
        self._env.discrete_action_input = True

        # :(
        self.academy_capabilities = UnityRLCapabilitiesProto()
        self.academy_capabilities.baseRLCapabilities = True
        self.academy_capabilities.concatenatedPngObservations = True
        self.academy_capabilities.compressedChannelMapping = True
        self.academy_capabilities.hybridActions = True
        self.academy_capabilities.trainingAnalytics = True
        self.academy_capabilities.variableLengthObservation = True
        self.academy_capabilities.multiAgentGroups = True
        self.count = 0
        self.episode_count = 0

    def step(self) -> None:
        if self._actions is None:
            self._obs, self._rew, self._done, _ = self._env.step([0] * self._env.n)
        else:
            self._obs, self._rew, self._done, _ = self._env.step(self._actions)
        if self.count % 25 == 0:
            self.reset()
            self._done = [True] * self._env.n
        self.count += 1
        if self.episode_count % 100 == 0:
            self._env.render(mode="agent")

    def reset(self) -> None:
        self._rew = [0] * self._env.n
        self._done = [False] * self._env.n
        self._actions = [0] * self._env.n
        self._obs = self._env.reset()
        self.episode_count += 1

    def close(self) -> None:
        self._env.close()

    @property
    def behavior_specs(self) -> MappingType[str, BehaviorSpec]:
        return {
            self._name: BehaviorSpec(
                [
                    ObservationSpec(
                        self._env.observation_space[0].shape,
                        (DimensionProperty.NONE,),
                        ObservationType.DEFAULT,
                        "obs_0",
                    )
                ],
                ActionSpec(0, (self._env.action_space[0].n,)),
            )
        }

    def set_actions(self, behavior_name: BehaviorName, action: ActionTuple) -> None:
        assert behavior_name == self._name
        self._actions = [action.discrete[i] for i in range(self._env.n)]

    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: ActionTuple
    ) -> None:
        assert behavior_name == self._name
        if self._actions is None:
            self._actions = [0] * self._env.n
        self._actions[agent_id] = action.discrete[0]

    def get_steps(
        self, behavior_name: BehaviorName
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        reward_scale = 1
        decision_obs = np.array(
            [self._obs[i] for i in range(self._env.n) if not self._done[i]],
            dtype=np.float32,
        )
        decision_reward = np.array(
            [
                self._rew[i] * 0
                for i in range(self._env.n)
                if not self._done[i]
            ],
            dtype=np.float32,
        )
        decision_id = np.array([i for i in range(self._env.n) if not self._done[i]])
        decision_group_reward = np.array(
            [self._rew[0] * reward_scale for i in range(self._env.n) if not self._done[i]],
            dtype=np.float32,
        )
        decision_group_id = np.array(
            [1 for i in range(self._env.n) if not self._done[i]]
        )
        decision_step = DecisionSteps(
            [decision_obs],
            decision_reward,
            decision_id,
            None,
            decision_group_id,
            decision_group_reward,
        )

        terminal_obs = np.array(
            [self._obs[i] for i in range(self._env.n) if self._done[i]],
            dtype=np.float32,
        )
        terminal_reward = np.array(
            [self._rew[i] * reward_scale for i in range(self._env.n) if self._done[i]],
            dtype=np.float32,
        )
        terminal_id = np.array([i for i in range(self._env.n) if self._done[i]])
        terminal_group_reward = np.array(
            [
                self._rew[0] * reward_scale
                for i in range(self._env.n)
                if self._done[i]
            ],
            dtype=np.float32,
        )
        terminal_group_id = np.array([1 for i in range(self._env.n) if self._done[i]])
        # TODO : Figureout the type of interruption
        terminal_interruption = np.array(
            [False for i in range(self._env.n) if self._done[i]]
        )
        terminal_steps = TerminalSteps(
            [terminal_obs],
            terminal_reward,
            terminal_interruption,
            terminal_id,
            terminal_group_id,
            terminal_group_reward,
        )

        return decision_step, terminal_steps

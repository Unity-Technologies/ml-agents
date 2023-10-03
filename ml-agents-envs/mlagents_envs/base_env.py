"""
Python Environment API for the ML-Agents Toolkit
The aim of this API is to expose Agents evolving in a simulation
to perform reinforcement learning on.
This API supports multi-agent scenarios and groups similar Agents (same
observations, actions spaces and behavior) together. These groups of Agents are
identified by their BehaviorName.
For performance reasons, the data of each group of agents is processed in a
batched manner. Agents are identified by a unique AgentId identifier that
allows tracking of Agents across simulation steps. Note that there is no
guarantee that the number or order of the Agents in the state will be
consistent across simulation steps.
A simulation steps corresponds to moving the simulation forward until at least
one agent in the simulation sends its observations to Python again. Since
Agents can request decisions at different frequencies, a simulation step does
not necessarily correspond to a fixed simulation time increment.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    List,
    NamedTuple,
    Tuple,
    Optional,
    Dict,
    Iterator,
    Any,
    Mapping as MappingType,
)
from enum import IntFlag, Enum
import numpy as np

from mlagents_envs.exception import UnityActionException

AgentId = int
GroupId = int
BehaviorName = str


class DecisionStep(NamedTuple):
    """
    Contains the data a single Agent collected since the last
    simulation step.
     - obs is a list of numpy arrays observations collected by the agent.
     - reward is a float. Corresponds to the rewards collected by the agent
     since the last simulation step.
     - agent_id is an int and an unique identifier for the corresponding Agent.
     - action_mask is an optional list of one dimensional array of booleans.
     Only available when using multi-discrete actions.
     Each array corresponds to an action branch. Each array contains a mask
     for each action of the branch. If true, the action is not available for
     the agent during this simulation step.
    """

    obs: List[np.ndarray]
    reward: float
    agent_id: AgentId
    action_mask: Optional[List[np.ndarray]]
    group_id: int
    group_reward: float


class DecisionSteps(Mapping):
    """
    Contains the data a batch of similar Agents collected since the last
    simulation step. Note that all Agents do not necessarily have new
    information to send at each simulation step. Therefore, the ordering of
    agents and the batch size of the DecisionSteps are not fixed across
    simulation steps.
     - obs is a list of numpy arrays observations collected by the batch of
     agent. Each obs has one extra dimension compared to DecisionStep: the
     first dimension of the array corresponds to the batch size of the batch.
     - reward is a float vector of length batch size. Corresponds to the
     rewards collected by each agent since the last simulation step.
     - agent_id is an int vector of length batch size containing unique
     identifier for the corresponding Agent. This is used to track Agents
     across simulation steps.
     - action_mask is an optional list of two dimensional array of booleans.
     Only available when using multi-discrete actions.
     Each array corresponds to an action branch. The first dimension of each
     array is the batch size and the second contains a mask for each action of
     the branch. If true, the action is not available for the agent during
     this simulation step.
    """

    def __init__(self, obs, reward, agent_id, action_mask, group_id, group_reward):
        self.obs: List[np.ndarray] = obs
        self.reward: np.ndarray = reward
        self.agent_id: np.ndarray = agent_id
        self.action_mask: Optional[List[np.ndarray]] = action_mask
        self.group_id: np.ndarray = group_id
        self.group_reward: np.ndarray = group_reward
        self._agent_id_to_index: Optional[Dict[AgentId, int]] = None

    @property
    def agent_id_to_index(self) -> Dict[AgentId, int]:
        """
        :returns: A Dict that maps agent_id to the index of those agents in
        this DecisionSteps.
        """
        if self._agent_id_to_index is None:
            self._agent_id_to_index = {}
            for a_idx, a_id in enumerate(self.agent_id):
                self._agent_id_to_index[a_id] = a_idx
        return self._agent_id_to_index

    def __len__(self) -> int:
        return len(self.agent_id)

    def __getitem__(self, agent_id: AgentId) -> DecisionStep:
        """
        returns the DecisionStep for a specific agent.
        :param agent_id: The id of the agent
        :returns: The DecisionStep
        """
        if agent_id not in self.agent_id_to_index:
            raise KeyError(f"agent_id {agent_id} is not present in the DecisionSteps")
        agent_index = self._agent_id_to_index[agent_id]  # type: ignore
        agent_obs = []
        for batched_obs in self.obs:
            agent_obs.append(batched_obs[agent_index])
        agent_mask = None
        if self.action_mask is not None:
            agent_mask = []
            for mask in self.action_mask:
                agent_mask.append(mask[agent_index])
        group_id = self.group_id[agent_index]
        return DecisionStep(
            obs=agent_obs,
            reward=self.reward[agent_index],
            agent_id=agent_id,
            action_mask=agent_mask,
            group_id=group_id,
            group_reward=self.group_reward[agent_index],
        )

    def __iter__(self) -> Iterator[Any]:
        yield from self.agent_id

    @staticmethod
    def empty(spec: "BehaviorSpec") -> "DecisionSteps":
        """
        Returns an empty DecisionSteps.
        :param spec: The BehaviorSpec for the DecisionSteps
        """
        obs: List[np.ndarray] = []
        for sen_spec in spec.observation_specs:
            obs += [np.zeros((0,) + sen_spec.shape, dtype=np.float32)]
        return DecisionSteps(
            obs=obs,
            reward=np.zeros(0, dtype=np.float32),
            agent_id=np.zeros(0, dtype=np.int32),
            action_mask=None,
            group_id=np.zeros(0, dtype=np.int32),
            group_reward=np.zeros(0, dtype=np.float32),
        )


class TerminalStep(NamedTuple):
    """
    Contains the data a single Agent collected when its episode ended.
     - obs is a list of numpy arrays observations collected by the agent.
     - reward is a float. Corresponds to the rewards collected by the agent
     since the last simulation step.
     - interrupted is a bool. Is true if the Agent was interrupted since the last
     decision step. For example, if the Agent reached the maximum number of steps for
     the episode.
     - agent_id is an int and an unique identifier for the corresponding Agent.
    """

    obs: List[np.ndarray]
    reward: float
    interrupted: bool
    agent_id: AgentId
    group_id: GroupId
    group_reward: float


class TerminalSteps(Mapping):
    """
    Contains the data a batch of Agents collected when their episode
    terminated. All Agents present in the TerminalSteps have ended their
    episode.
     - obs is a list of numpy arrays observations collected by the batch of
     agent. Each obs has one extra dimension compared to DecisionStep: the
     first dimension of the array corresponds to the batch size of the batch.
     - reward is a float vector of length batch size. Corresponds to the
     rewards collected by each agent since the last simulation step.
     - interrupted is an array of booleans of length batch size. Is true if the
     associated Agent was interrupted since the last decision step. For example, if the
     Agent reached the maximum number of steps for the episode.
     - agent_id is an int vector of length batch size containing unique
     identifier for the corresponding Agent. This is used to track Agents
     across simulation steps.
    """

    def __init__(self, obs, reward, interrupted, agent_id, group_id, group_reward):
        self.obs: List[np.ndarray] = obs
        self.reward: np.ndarray = reward
        self.interrupted: np.ndarray = interrupted
        self.agent_id: np.ndarray = agent_id
        self.group_id: np.ndarray = group_id
        self.group_reward: np.ndarray = group_reward
        self._agent_id_to_index: Optional[Dict[AgentId, int]] = None

    @property
    def agent_id_to_index(self) -> Dict[AgentId, int]:
        """
        :returns: A Dict that maps agent_id to the index of those agents in
        this TerminalSteps.
        """
        if self._agent_id_to_index is None:
            self._agent_id_to_index = {}
            for a_idx, a_id in enumerate(self.agent_id):
                self._agent_id_to_index[a_id] = a_idx
        return self._agent_id_to_index

    def __len__(self) -> int:
        return len(self.agent_id)

    def __getitem__(self, agent_id: AgentId) -> TerminalStep:
        """
        returns the TerminalStep for a specific agent.
        :param agent_id: The id of the agent
        :returns: obs, reward, done, agent_id and optional action mask for a
        specific agent
        """
        if agent_id not in self.agent_id_to_index:
            raise KeyError(f"agent_id {agent_id} is not present in the TerminalSteps")
        agent_index = self._agent_id_to_index[agent_id]  # type: ignore
        agent_obs = []
        for batched_obs in self.obs:
            agent_obs.append(batched_obs[agent_index])
        group_id = self.group_id[agent_index]
        return TerminalStep(
            obs=agent_obs,
            reward=self.reward[agent_index],
            interrupted=self.interrupted[agent_index],
            agent_id=agent_id,
            group_id=group_id,
            group_reward=self.group_reward[agent_index],
        )

    def __iter__(self) -> Iterator[Any]:
        yield from self.agent_id

    @staticmethod
    def empty(spec: "BehaviorSpec") -> "TerminalSteps":
        """
        Returns an empty TerminalSteps.
        :param spec: The BehaviorSpec for the TerminalSteps
        """
        obs: List[np.ndarray] = []
        for sen_spec in spec.observation_specs:
            obs += [np.zeros((0,) + sen_spec.shape, dtype=np.float32)]
        return TerminalSteps(
            obs=obs,
            reward=np.zeros(0, dtype=np.float32),
            interrupted=np.zeros(0, dtype=bool),
            agent_id=np.zeros(0, dtype=np.int32),
            group_id=np.zeros(0, dtype=np.int32),
            group_reward=np.zeros(0, dtype=np.float32),
        )


class _ActionTupleBase(ABC):
    """
    An object whose fields correspond to action data of continuous and discrete
    spaces. Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size),
    respectively. Note, this also holds when continuous or discrete size is
    zero.
    """

    def __init__(
        self,
        continuous: Optional[np.ndarray] = None,
        discrete: Optional[np.ndarray] = None,
    ):
        self._continuous: Optional[np.ndarray] = None
        self._discrete: Optional[np.ndarray] = None
        if continuous is not None:
            self.add_continuous(continuous)
        if discrete is not None:
            self.add_discrete(discrete)

    @property
    def continuous(self) -> np.ndarray:
        return self._continuous

    @property
    def discrete(self) -> np.ndarray:
        return self._discrete

    def add_continuous(self, continuous: np.ndarray) -> None:
        if continuous.dtype != np.float32:
            continuous = continuous.astype(np.float32, copy=False)
        if self._discrete is None:
            self._discrete = np.zeros(
                (continuous.shape[0], 0), dtype=self.discrete_dtype
            )
        self._continuous = continuous

    def add_discrete(self, discrete: np.ndarray) -> None:
        if discrete.dtype != self.discrete_dtype:
            discrete = discrete.astype(self.discrete_dtype, copy=False)
        if self._continuous is None:
            self._continuous = np.zeros((discrete.shape[0], 0), dtype=np.float32)
        self._discrete = discrete

    @property
    @abstractmethod
    def discrete_dtype(self) -> np.dtype:
        pass


class ActionTuple(_ActionTupleBase):
    """
    An object whose fields correspond to actions of different types.
    Continuous and discrete actions are numpy arrays of type float32 and
    int32, respectively and are type checked on construction.
    Dimensions are of (n_agents, continuous_size) and (n_agents, discrete_size),
    respectively. Note, this also holds when continuous or discrete size is
    zero.
    """

    @property
    def discrete_dtype(self) -> np.dtype:
        """
        The dtype of a discrete action.
        """
        return np.int32


class ActionSpec(NamedTuple):
    """
    A NamedTuple containing utility functions and information about the action spaces
    for a group of Agents under the same behavior.
    - num_continuous_actions is an int corresponding to the number of floats which
    constitute the action.
    - discrete_branch_sizes is a Tuple of int where each int corresponds to
    the number of discrete actions available to the agent on an independent action branch.
    """

    continuous_size: int
    discrete_branches: Tuple[int, ...]

    def __eq__(self, other):
        return (
            self.continuous_size == other.continuous_size
            and self.discrete_branches == other.discrete_branches
        )

    def __str__(self):
        return f"Continuous: {self.continuous_size}, Discrete: {self.discrete_branches}"

    # For backwards compatibility
    def is_discrete(self) -> bool:
        """
        Returns true if this Behavior uses discrete actions
        """
        return self.discrete_size > 0 and self.continuous_size == 0

    # For backwards compatibility
    def is_continuous(self) -> bool:
        """
        Returns true if this Behavior uses continuous actions
        """
        return self.discrete_size == 0 and self.continuous_size > 0

    @property
    def discrete_size(self) -> int:
        """
        Returns a an int corresponding to the number of discrete branches.
        """
        return len(self.discrete_branches)

    def empty_action(self, n_agents: int) -> ActionTuple:
        """
        Generates ActionTuple corresponding to an empty action (all zeros)
        for a number of agents.
        :param n_agents: The number of agents that will have actions generated
        """
        _continuous = np.zeros((n_agents, self.continuous_size), dtype=np.float32)
        _discrete = np.zeros((n_agents, self.discrete_size), dtype=np.int32)
        return ActionTuple(continuous=_continuous, discrete=_discrete)

    def random_action(self, n_agents: int) -> ActionTuple:
        """
        Generates ActionTuple corresponding to a random action (either discrete
        or continuous) for a number of agents.
        :param n_agents: The number of agents that will have actions generated
        """
        _continuous = np.random.uniform(
            low=-1.0, high=1.0, size=(n_agents, self.continuous_size)
        )
        _discrete = np.zeros((n_agents, self.discrete_size), dtype=np.int32)
        if self.discrete_size > 0:
            _discrete = np.column_stack(
                [
                    np.random.randint(
                        0,
                        self.discrete_branches[i],  # type: ignore
                        size=(n_agents),
                        dtype=np.int32,
                    )
                    for i in range(self.discrete_size)
                ]
            )
        return ActionTuple(continuous=_continuous, discrete=_discrete)

    def _validate_action(
        self, actions: ActionTuple, n_agents: int, name: str
    ) -> ActionTuple:
        """
        Validates that action has the correct action dim
        for the correct number of agents and ensures the type.
        """
        _expected_shape = (n_agents, self.continuous_size)
        if actions.continuous.shape != _expected_shape:
            raise UnityActionException(
                f"The behavior {name} needs a continuous input of dimension "
                f"{_expected_shape} for (<number of agents>, <action size>) but "
                f"received input of dimension {actions.continuous.shape}"
            )
        _expected_shape = (n_agents, self.discrete_size)
        if actions.discrete.shape != _expected_shape:
            raise UnityActionException(
                f"The behavior {name} needs a discrete input of dimension "
                f"{_expected_shape} for (<number of agents>, <action size>) but "
                f"received input of dimension {actions.discrete.shape}"
            )
        return actions

    @staticmethod
    def create_continuous(continuous_size: int) -> "ActionSpec":
        """
        Creates an ActionSpec that is homogenously continuous
        """
        return ActionSpec(continuous_size, ())

    @staticmethod
    def create_discrete(discrete_branches: Tuple[int]) -> "ActionSpec":
        """
        Creates an ActionSpec that is homogenously discrete
        """
        return ActionSpec(0, discrete_branches)

    @staticmethod
    def create_hybrid(
        continuous_size: int, discrete_branches: Tuple[int]
    ) -> "ActionSpec":
        """
        Creates a hybrid ActionSpace
        """
        return ActionSpec(continuous_size, discrete_branches)


class DimensionProperty(IntFlag):
    """
    The dimension property of a dimension of an observation.
    """

    UNSPECIFIED = 0
    """
    No properties specified.
    """

    NONE = 1
    """
    No Property of the observation in that dimension. Observation can be processed with
    Fully connected networks.
    """

    TRANSLATIONAL_EQUIVARIANCE = 2
    """
    Means it is suitable to do a convolution in this dimension.
    """

    VARIABLE_SIZE = 4
    """
    Means that there can be a variable number of observations in this dimension.
    The observations are unordered.
    """


class ObservationType(Enum):
    """
    An Enum which defines the type of information carried in the observation
    of the agent.
    """

    DEFAULT = 0
    """
    Observation information is generic.
    """

    GOAL_SIGNAL = 1
    """
    Observation contains goal information for current task.
    """


class ObservationSpec(NamedTuple):
    """
    A NamedTuple containing information about the observation of Agents.
    - shape is a Tuple of int : It corresponds to the shape of
    an observation's dimensions.
    - dimension_property is a Tuple of DimensionProperties flag, one flag for each
    dimension.
    - observation_type is an enum of ObservationType.
    """

    shape: Tuple[int, ...]
    dimension_property: Tuple[DimensionProperty, ...]
    observation_type: ObservationType

    # Optional name. For observations coming from com.unity.ml-agents, this
    # will be the ISensor name.
    name: str


class BehaviorSpec(NamedTuple):
    """
    A NamedTuple containing information about the observation and action
    spaces for a group of Agents under the same behavior.
    - observation_specs is a List of ObservationSpec NamedTuple containing
    information about the information of the Agent's observations such as their shapes.
    The order of the ObservationSpec is the same as the order of the observations of an
    agent.
    - action_spec is an ActionSpec NamedTuple.
    """

    observation_specs: List[ObservationSpec]
    action_spec: ActionSpec


class BehaviorMapping(Mapping):
    def __init__(self, specs: Dict[BehaviorName, BehaviorSpec]):
        self._dict = specs

    def __len__(self) -> int:
        return len(self._dict)

    def __getitem__(self, behavior: BehaviorName) -> BehaviorSpec:
        return self._dict[behavior]

    def __iter__(self) -> Iterator[Any]:
        yield from self._dict


class BaseEnv(ABC):
    @abstractmethod
    def step(self) -> None:
        """
        Signals the environment that it must move the simulation forward
        by one step.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Signals the environment that it must reset the simulation.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Signals the environment that it must close.
        """

    @property
    @abstractmethod
    def behavior_specs(self) -> MappingType[str, BehaviorSpec]:
        """
        Returns a Mapping from behavior names to behavior specs.
        Agents grouped under the same behavior name have the same action and
        observation specs, and are expected to behave similarly in the
        environment.
        Note that new keys can be added to this mapping as new policies are instantiated.
        """

    @abstractmethod
    def set_actions(self, behavior_name: BehaviorName, action: ActionTuple) -> None:
        """
        Sets the action for all of the agents in the simulation for the next
        step. The Actions must be in the same order as the order received in
        the DecisionSteps.
        :param behavior_name: The name of the behavior the agents are part of
        :param action: ActionTuple tuple of continuous and/or discrete action.
        Actions are np.arrays with dimensions  (n_agents, continuous_size) and
        (n_agents, discrete_size), respectively.
        """

    @abstractmethod
    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: ActionTuple
    ) -> None:
        """
        Sets the action for one of the agents in the simulation for the next
        step.
        :param behavior_name: The name of the behavior the agent is part of
        :param agent_id: The id of the agent the action is set for
        :param action: ActionTuple tuple of continuous and/or discrete action
        Actions are np.arrays with dimensions  (1, continuous_size) and
        (1, discrete_size), respectively. Note, this initial dimensions of 1 is because
        this action is meant for a single agent.
        """

    @abstractmethod
    def get_steps(
        self, behavior_name: BehaviorName
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        """
        Retrieves the steps of the agents that requested a step in the
        simulation.
        :param behavior_name: The name of the behavior the agents are part of
        :return: A tuple containing :
         - A DecisionSteps NamedTuple containing the observations,
         the rewards, the agent ids and the action masks for the Agents
         of the specified behavior. These Agents need an action this step.
         - A TerminalSteps NamedTuple containing the observations,
         rewards, agent ids and interrupted flags of the agents that had their
         episode terminated last step.
        """

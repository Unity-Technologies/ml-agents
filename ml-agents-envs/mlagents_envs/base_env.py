"""
Python Environment API for the ML-Agents toolkit
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
    Union,
    Dict,
    Iterator,
    Any,
    Mapping as MappingType,
)
import numpy as np
from enum import Enum

AgentId = int
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
     Only available in multi-discrete action space type.
     Each array corresponds to an action branch. Each array contains a mask
     for each action of the branch. If true, the action is not available for
     the agent during this simulation step.
    """

    obs: List[np.ndarray]
    reward: float
    agent_id: AgentId
    action_mask: Optional[List[np.ndarray]]


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
     Only available in multi-discrete action space type.
     Each array corresponds to an action branch. The first dimension of each
     array is the batch size and the second contains a mask for each action of
     the branch. If true, the action is not available for the agent during
     this simulation step.
    """

    def __init__(self, obs, reward, agent_id, action_mask):
        self.obs: List[np.ndarray] = obs
        self.reward: np.ndarray = reward
        self.agent_id: np.ndarray = agent_id
        self.action_mask: Optional[List[np.ndarray]] = action_mask
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
            raise KeyError(
                "agent_id {} is not present in the DecisionSteps".format(agent_id)
            )
        agent_index = self._agent_id_to_index[agent_id]  # type: ignore
        agent_obs = []
        for batched_obs in self.obs:
            agent_obs.append(batched_obs[agent_index])
        agent_mask = None
        if self.action_mask is not None:
            agent_mask = []
            for mask in self.action_mask:
                agent_mask.append(mask[agent_index])
        return DecisionStep(
            obs=agent_obs,
            reward=self.reward[agent_index],
            agent_id=agent_id,
            action_mask=agent_mask,
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
        for shape in spec.observation_shapes:
            obs += [np.zeros((0,) + shape, dtype=np.float32)]
        return DecisionSteps(
            obs=obs,
            reward=np.zeros(0, dtype=np.float32),
            agent_id=np.zeros(0, dtype=np.int32),
            action_mask=None,
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

    def __init__(self, obs, reward, interrupted, agent_id):
        self.obs: List[np.ndarray] = obs
        self.reward: np.ndarray = reward
        self.interrupted: np.ndarray = interrupted
        self.agent_id: np.ndarray = agent_id
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
            raise KeyError(
                "agent_id {} is not present in the TerminalSteps".format(agent_id)
            )
        agent_index = self._agent_id_to_index[agent_id]  # type: ignore
        agent_obs = []
        for batched_obs in self.obs:
            agent_obs.append(batched_obs[agent_index])
        return TerminalStep(
            obs=agent_obs,
            reward=self.reward[agent_index],
            interrupted=self.interrupted[agent_index],
            agent_id=agent_id,
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
        for shape in spec.observation_shapes:
            obs += [np.zeros((0,) + shape, dtype=np.float32)]
        return TerminalSteps(
            obs=obs,
            reward=np.zeros(0, dtype=np.float32),
            interrupted=np.zeros(0, dtype=np.bool),
            agent_id=np.zeros(0, dtype=np.int32),
        )


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class BehaviorSpec(NamedTuple):
    """
    A NamedTuple to containing information about the observations and actions
    spaces for a group of Agents under the same behavior.
     - observation_shapes is a List of Tuples of int : Each Tuple corresponds
     to an observation's dimensions. The shape tuples have the same ordering as
     the ordering of the DecisionSteps and TerminalSteps.
     - action_type is the type of data of the action. it can be discrete or
     continuous. If discrete, the action tensors are expected to be int32. If
     continuous, the actions are expected to be float32.
     - action_shape is:
       - An int in continuous action space corresponding to the number of
     floats that constitute the action.
       - A Tuple of int in discrete action space where each int corresponds to
       the number of discrete actions available to the agent.
    """

    observation_shapes: List[Tuple]
    action_type: ActionType
    action_shape: Union[int, Tuple[int, ...]]

    def is_action_discrete(self) -> bool:
        """
        Returns true if this Behavior uses discrete actions
        """
        return self.action_type == ActionType.DISCRETE

    def is_action_continuous(self) -> bool:
        """
        Returns true if this Behavior uses continuous actions
        """
        return self.action_type == ActionType.CONTINUOUS

    @property
    def action_size(self) -> int:
        """
        Returns the dimension of the action.
         - In the continuous case, will return the number of continuous actions.
         - In the (multi-)discrete case, will return the number of action.
         branches.
        """
        if self.action_type == ActionType.DISCRETE:
            return len(self.action_shape)  # type: ignore
        else:
            return self.action_shape  # type: ignore

    @property
    def discrete_action_branches(self) -> Optional[Tuple[int, ...]]:
        """
        Returns a Tuple of int corresponding to the number of possible actions
        for each branch (only for discrete actions). Will return None in
        for continuous actions.
        """
        if self.action_type == ActionType.DISCRETE:
            return self.action_shape  # type: ignore
        else:
            return None

    def create_empty_action(self, n_agents: int) -> np.ndarray:
        if self.action_type == ActionType.DISCRETE:
            return np.zeros((n_agents, self.action_size), dtype=np.int32)
        else:
            return np.zeros((n_agents, self.action_size), dtype=np.float32)


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
    def set_actions(self, behavior_name: BehaviorName, action: np.ndarray) -> None:
        """
        Sets the action for all of the agents in the simulation for the next
        step. The Actions must be in the same order as the order received in
        the DecisionSteps.
        :param behavior_name: The name of the behavior the agents are part of
        :param action: A two dimensional np.ndarray corresponding to the action
        (either int or float)
        """

    @abstractmethod
    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: np.ndarray
    ) -> None:
        """
        Sets the action for one of the agents in the simulation for the next
        step.
        :param behavior_name: The name of the behavior the agent is part of
        :param agent_id: The id of the agent the action is set for
        :param action: A one dimensional np.ndarray corresponding to the action
        (either int or float)
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

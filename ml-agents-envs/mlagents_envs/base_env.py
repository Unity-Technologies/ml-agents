"""
Python Environment API for the ML-Agents toolkit
The aim of this API is to expose groups of similar Agents evolving in Unity
to perform reinforcement learning on.
There can be multiple groups of similar Agents (same observations and actions
spaces) in the simulation. These groups are identified by a agent_group that
corresponds to a single group of Agents in the simulation.
For performance reasons, the data of each group of agents is processed in a
batched manner. When retrieving the state of a group of Agents, said state
contains the data for the whole group. Agents in these groups are identified
by a unique int identifier that allows tracking of Agents across simulation
steps. Note that there is no guarantee that the number or order of the Agents
in the state will be consistent across simulation steps.
A simulation steps corresponds to moving the simulation forward until at least
one agent in the simulation sends its observations to Python again. Since
Agents can request decisions at different frequencies, a simulation step does
not necessarily correspond to a fixed simulation time increment.
"""

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple, Optional, Union, Dict
import numpy as np
from enum import Enum

AgentId = int
AgentGroup = str


class StepResult(NamedTuple):
    """
    Contains the data a single Agent collected since the last
    simulation step.
     - obs is a list of numpy arrays observations collected by the group of
     agent.
     - reward is a float. Corresponds to the rewards collected by the agent
     since the last simulation step.
     - done is a bool. Is true if the Agent was terminated during the last
     simulation step.
     - max_step is a bool. Is true if the Agent reached its maximum number of
     steps during the last simulation step.
     - agent_id is an int and an unique identifier for the corresponding Agent.
     - action_mask is an optional list of one dimensional array of booleans.
     Only available in multi-discrete action space type.
     Each array corresponds to an action branch. Each array contains a mask
     for each action of the branch. If true, the action is not available for
     the agent during this simulation step.
    """

    obs: List[np.ndarray]
    reward: float
    done: bool
    max_step: bool
    agent_id: AgentId
    action_mask: Optional[List[np.ndarray]]


class BatchedStepResult:
    """
    Contains the data a group of similar Agents collected since the last
    simulation step. Note that all Agents do not necessarily have new
    information to send at each simulation step. Therefore, the ordering of
    agents and the batch size of the BatchedStepResult are not fixed across
    simulation steps.
     - obs is a list of numpy arrays observations collected by the group of
     agent. Each obs has one extra dimension compared to StepResult: the first
     dimension of the array corresponds to the batch size of
     the group.
     - reward is a float vector of length batch size. Corresponds to the
     rewards collected by each agent since the last simulation step.
     - done is an array of booleans of length batch size. Is true if the
     associated Agent was terminated during the last simulation step.
     - max_step is an array of booleans of length batch size. Is true if the
     associated Agent reached its maximum number of steps during the last
     simulation step.
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

    def __init__(self, obs, reward, done, max_step, agent_id, action_mask):
        self.obs: List[np.ndarray] = obs
        self.reward: np.ndarray = reward
        self.done: np.ndarray = done
        self.max_step: np.ndarray = max_step
        self.agent_id: np.ndarray = agent_id
        self.action_mask: Optional[List[np.ndarray]] = action_mask
        self._agent_id_to_index: Optional[Dict[AgentId, int]] = None

    @property
    def agent_id_to_index(self) -> Dict[AgentId, int]:
        """
        Returns the index of the agent_id in this BatchedStepResult, and
        -1 if agent_id is not in this BatchedStepResult.
        :param agent_id: The id of the agent
        :returns: The index of the agent_id, and -1 if not found.
        """
        if self._agent_id_to_index is None:
            self._agent_id_to_index = {}
            for a_idx, a_id in enumerate(self.agent_id):
                self._agent_id_to_index[a_id] = a_idx
        return self._agent_id_to_index

    def contains_agent(self, agent_id: AgentId) -> bool:
        return agent_id in self.agent_id_to_index

    def get_agent_step_result(self, agent_id: AgentId) -> StepResult:
        """
        returns the step result for a specific agent.
        :param agent_id: The id of the agent
        :returns: obs, reward, done, agent_id and optional action mask for a
        specific agent
        """
        if not self.contains_agent(agent_id):
            raise IndexError(
                "agent_id {} is not present in the BatchedStepResult".format(agent_id)
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
        return StepResult(
            obs=agent_obs,
            reward=self.reward[agent_index],
            done=self.done[agent_index],
            max_step=self.max_step[agent_index],
            agent_id=agent_id,
            action_mask=agent_mask,
        )

    @staticmethod
    def empty(spec: "AgentGroupSpec") -> "BatchedStepResult":
        """
        Returns an empty BatchedStepResult.
        :param spec: The AgentGroupSpec for the BatchedStepResult
        """
        obs: List[np.ndarray] = []
        for shape in spec.observation_shapes:
            obs += [np.zeros((0,) + shape, dtype=np.float32)]
        return BatchedStepResult(
            obs=obs,
            reward=np.zeros(0, dtype=np.float32),
            done=np.zeros(0, dtype=np.bool),
            max_step=np.zeros(0, dtype=np.bool),
            agent_id=np.zeros(0, dtype=np.int32),
            action_mask=None,
        )

    def n_agents(self) -> int:
        return len(self.agent_id)


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class AgentGroupSpec(NamedTuple):
    """
    A NamedTuple to containing information about the observations and actions
    spaces for a group of Agents.
     - observation_shapes is a List of Tuples of int : Each Tuple corresponds
     to an observation's dimensions. The shape tuples have the same ordering as
     the ordering of the BatchedStepResult and StepResult.
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
        Returns true if the Agent group uses discrete actions
        """
        return self.action_type == ActionType.DISCRETE

    def is_action_continuous(self) -> bool:
        """
        Returns true if the Agent group uses continuous actions
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


class BaseEnv(ABC):
    @abstractmethod
    def step(self) -> None:
        """
        Signals the environment that it must move the simulation forward
        by one step.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Signals the environment that it must reset the simulation.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Signals the environment that it must close.
        """
        pass

    @abstractmethod
    def get_agent_groups(self) -> List[AgentGroup]:
        """
        Returns the list of the agent group names present in the environment.
        Agents grouped under the same group name have the same action and
        observation specs, and are expected to behave similarly in the environment.
        This list can grow with time as new policies are instantiated.
        :return: the list of agent group names.
        """
        pass

    @abstractmethod
    def set_actions(self, agent_group: AgentGroup, action: np.ndarray) -> None:
        """
        Sets the action for all of the agents in the simulation for the next
        step. The Actions must be in the same order as the order received in
        the step result.
        :param agent_group: The name of the group the agents are part of
        :param action: A two dimensional np.ndarray corresponding to the action
        (either int or float)
        """
        pass

    @abstractmethod
    def set_action_for_agent(
        self, agent_group: AgentGroup, agent_id: AgentId, action: np.ndarray
    ) -> None:
        """
        Sets the action for one of the agents in the simulation for the next
        step.
        :param agent_group: The name of the group the agent is part of
        :param agent_id: The id of the agent the action is set for
        :param action: A two dimensional np.ndarray corresponding to the action
        (either int or float)
        """
        pass

    @abstractmethod
    def get_step_result(self, agent_group: AgentGroup) -> BatchedStepResult:
        """
        Retrieves the observations of the agents that requested a step in the
        simulation.
        :param agent_group: The name of the group the agents are part of
        :return: A BatchedStepResult NamedTuple containing the observations,
        the rewards and the done flags for this group of agents.
        """
        pass

    @abstractmethod
    def get_agent_group_spec(self, agent_group: AgentGroup) -> AgentGroupSpec:
        """
        Get the AgentGroupSpec corresponding to the agent group name
        :param agent_group: The name of the group the agents are part of
        :return: A AgentGroupSpec corresponding to that agent group name
        """
        pass

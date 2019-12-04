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
by a unique int identifier that allows tracking of Agents accross simulation
steps. Note that there is no guarantee that the number or order of the Agents
in the state will be consistent accross simulation steps.
A simulation steps corresponds to moving the simulation forward until at least
one agent in the simulation sends its observations to Python again. Since
Agents can request decisions at different frequencies, a simulation step does
not necessarily correspond to a fixed simulation time increment.
Changes from ML-Agents v0.11 :
 - Step now takes no arguments and returns None.
 - The data received from a step is no longer a dictionary of NamedTuple, the
state data must now be retrieved manually from the environment object.
 - Reset does no longer take any input arguments.
 - Modifying properties of the environment besides actions is handled in
SideChannels.
"""

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple, Union, Optional
import numpy as np
from enum import Enum


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
     Each array corresponds to an action branch. Each array contains a mask
     for each action of the branch. If true, the action is not available for
     the agent during this simulation step.
    """

    obs: List[np.array]
    reward: float
    done: bool
    max_step: bool
    agent_id: int
    action_mask: Optional[List[np.array]]


class BatchedStepResult(NamedTuple):
    """
    Contains the data a group of similar Agents collected since the last
    simulation step. Note that all Agents do not necessarily have new
    information to send at each simulation step. Therefore, the ordering of
    agents and the batch size of the BatchedStepResult are not fixed accross
    simulation steps.
     - obs is a list of numpy arrays observations collected by the group of
    agent. The first dimension of the array corresponds to the batch size of
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
     accross simulation steps.
     - action_mask is an optional list of two dimensional array of booleans.
     Each array corresponds to an action branch. The first dimension of each
     array is the batch size and the second contains a mask for each action of
     the branch. If true, the action is not available for the agent during
     this simulation step.
    """

    obs: List[np.array]
    reward: np.array
    done: np.array
    max_step: np.array
    agent_id: np.array
    action_mask: Optional[List[np.array]]

    def get_agent_step_result(self, agent_id: int) -> StepResult:
        """
        returns the step result for a specific agent.
        :param agent_id: The id of the agent
        :returns: obs, reward, done, agent_id and optional action mask for a
        specific agent
        """
        try:
            agent_index = np.where(self.agent_id == agent_id)[0][0]
        except IndexError as ie:
            raise IndexError(
                "agent_id {} is not present in the BatchedStepResult".format(agent_id)
            ) from ie
        agent_obs = []
        for batched_obs in self.obs:
            agent_obs.append(batched_obs[agent_index])
        agent_mask = None
        if self.action_mask is not None:
            agent_mask = []
            for mask in self.action_mask:
                agent_mask.append(mask[0])
        return StepResult(
            obs=agent_obs,
            reward=self.reward[agent_index],
            done=self.done[agent_index],
            max_step=self.max_step[agent_index],
            agent_id=agent_id,
            action_mask=agent_mask,
        )

    @staticmethod
    def empty(spec):
        """
        Returns an empty BatchedStepResult.
        :param spec: The AgentGroupSpec for the BatchedStepResult
        """
        obs = []
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
     to an observation's dimensionsthe shape tuples have the same ordering as
     the ordering of the BatchedStepResult.
     - action_type is the type of data of the action. it can be discrete or
     continuous. If discrete, the action tensors are expected to be int32. If
     discrete, the actions are expected to be float32.
     - action_shape is:
       - An int in continuous action space corresponding to the number of
     floats that constitute the action.
       - A Tuple of int in discrete action space where each int corresponds to
       the number of discrete actions available to the agent.
    """

    observation_shapes: List[Tuple]
    action_type: ActionType
    action_shape: Union[int, Tuple]


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
    def get_agent_groups(self) -> List[str]:
        """
        Returns the list of the agent group names present in the environment.
        Agents grouped under the same group name have the same action and
        observation specs are expected to behave similarly in the environment.
        This list can grow with time as new policies are instantiated.
        :return: the list of agent group names.
        """
        pass

    @abstractmethod
    def set_action(self, agent_group: str, action: np.array) -> None:
        """
        Sets the action for all of the agents in the simulation for the next
        step. The Actions must be in the same order as the order received in
        the step result.
        :param agent_group: The name of the group the agents are part of
        :param action: A two dimensional np.array corresponding to the action
        (either int or float)
        """
        pass

    @abstractmethod
    def set_action_for_agent(
        self, agent_group: str, agent_id: int, action: np.array
    ) -> None:
        """
        Sets the action for one of the agents in the simulation for the next
        step.
        :param agent_group: The name of the group the agent is part of
        :param agent_id: The id of the agent the action is set for
        :param action: A two dimensional np.array corresponding to the action
        (either int or float)
        """
        pass

    @abstractmethod
    def get_step_result(self, agent_group: str) -> BatchedStepResult:
        """
        Retrieves the observations of the agents that requested a step in the
        simulation.
        :param agent_group: The name of the group the agents are part of
        :return: A BatchedStepResult NamedTuple containing the observations,
        the rewards and the done flags for this group of agents.
        """
        pass

    @abstractmethod
    def get_agent_group_spec(self, agent_group: str) -> AgentGroupSpec:
        """
        Get the AgentGroupSpec corresponding to the agent group name
        :param agent_group: The name of the group the agents are part of
        :return: A AgentGroupSpec corresponding to that agent group name
        """
        pass

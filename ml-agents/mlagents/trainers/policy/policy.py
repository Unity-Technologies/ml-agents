from abc import abstractmethod
from typing import Dict, List, Optional
import numpy as np

from mlagents_envs.base_env import DecisionSteps
from mlagents_envs.exception import UnityException

from mlagents.trainers.action_info import ActionInfo


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Policy(object):
    def __init__(self, brain, seed, trainer_params):
        self.brain = brain
        self.seed = seed
        self.model_path = None
        self.use_continuous_act = brain.vector_action_space_type == "continuous"
        if self.use_continuous_act:
            self.num_branches = self.brain.vector_action_space_size[0]
        else:
            self.num_branches = len(self.brain.vector_action_space_size)
        self.previous_action_dict: Dict[str, np.array] = {}
        self.memory_dict: Dict[str, np.ndarray] = {}
        self.normalize = trainer_params["normalize"]
        self.use_recurrent = trainer_params["use_recurrent"]

        if self.use_recurrent:
            self.m_size = trainer_params["memory_size"]
            self.sequence_length = trainer_params["sequence_length"]
            if self.m_size == 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is 0 even "
                    "though the trainer uses recurrent.".format(brain.brain_name)
                )
            elif self.m_size % 2 != 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is {1} "
                    "but it must be divisible by 2.".format(
                        brain.brain_name, self.m_size
                    )
                )

    def make_empty_memory(self, num_agents):
        """
        Creates empty memory for use with RNNs
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.m_size), dtype=np.float32)

    def save_memories(
        self, agent_ids: List[str], memory_matrix: Optional[np.ndarray]
    ) -> None:
        if memory_matrix is None:
            return
        for index, agent_id in enumerate(agent_ids):
            self.memory_dict[agent_id] = memory_matrix[index, :]

    def retrieve_memories(self, agent_ids: List[str]) -> np.ndarray:
        memory_matrix = np.zeros((len(agent_ids), self.m_size), dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.memory_dict:
                memory_matrix[index, :] = self.memory_dict[agent_id]
        return memory_matrix

    def remove_memories(self, agent_ids):
        for agent_id in agent_ids:
            if agent_id in self.memory_dict:
                self.memory_dict.pop(agent_id)

    def make_empty_previous_action(self, num_agents):
        """
        Creates empty previous action for use with RNNs and discrete control
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.num_branches), dtype=np.int)

    def save_previous_action(
        self, agent_ids: List[str], action_matrix: Optional[np.ndarray]
    ) -> None:
        if action_matrix is None:
            return
        for index, agent_id in enumerate(agent_ids):
            self.previous_action_dict[agent_id] = action_matrix[index, :]

    def retrieve_previous_action(self, agent_ids: List[str]) -> np.ndarray:
        action_matrix = np.zeros((len(agent_ids), self.num_branches), dtype=np.int)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.previous_action_dict:
                action_matrix[index, :] = self.previous_action_dict[agent_id]
        return action_matrix

    def remove_previous_action(self, agent_ids):
        for agent_id in agent_ids:
            if agent_id in self.previous_action_dict:
                self.previous_action_dict.pop(agent_id)

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        raise NotImplementedError

    @abstractmethod
    def increment_step(self, n_steps):
        pass

    @abstractmethod
    def save_model(self, step):
        pass

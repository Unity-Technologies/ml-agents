from abc import abstractmethod
from typing import Dict, List, Optional
import numpy as np

from mlagents_envs.base_env import ActionTuple, BehaviorSpec, DecisionSteps
from mlagents_envs.exception import UnityException

from mlagents.trainers.action_info import ActionInfo
from mlagents.trainers.settings import TrainerSettings, NetworkSettings
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.behavior_id_utils import GlobalAgentId


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class Policy:
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        trainer_settings: TrainerSettings,
        tanh_squash: bool = False,
        condition_sigma_on_obs: bool = True,
    ):
        self.behavior_spec = behavior_spec
        self.trainer_settings = trainer_settings
        self.network_settings: NetworkSettings = trainer_settings.network_settings
        self.seed = seed
        self.previous_action_dict: Dict[str, np.ndarray] = {}
        self.previous_memory_dict: Dict[str, np.ndarray] = {}
        self.memory_dict: Dict[str, np.ndarray] = {}
        self.normalize = trainer_settings.network_settings.normalize
        self.use_recurrent = self.network_settings.memory is not None
        self.h_size = self.network_settings.hidden_units
        num_layers = self.network_settings.num_layers
        if num_layers < 1:
            num_layers = 1
        self.num_layers = num_layers

        self.vis_encode_type = self.network_settings.vis_encode_type
        self.tanh_squash = tanh_squash
        self.condition_sigma_on_obs = condition_sigma_on_obs

        self.m_size = 0
        self.sequence_length = 1
        if self.network_settings.memory is not None:
            self.m_size = self.network_settings.memory.memory_size
            self.sequence_length = self.network_settings.memory.sequence_length

        # Non-exposed parameters; these aren't exposed because they don't have a
        # good explanation and usually shouldn't be touched.
        self.log_std_min = -20
        self.log_std_max = 2

    def make_empty_memory(self, num_agents):
        """
        Creates empty memory for use with RNNs
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.m_size), dtype=np.float32)

    def save_memories(
        self, agent_ids: List[GlobalAgentId], memory_matrix: Optional[np.ndarray]
    ) -> None:
        if memory_matrix is None:
            return

        # Pass old memories into previous_memory_dict
        for agent_id in agent_ids:
            if agent_id in self.memory_dict:
                self.previous_memory_dict[agent_id] = self.memory_dict[agent_id]

        for index, agent_id in enumerate(agent_ids):
            self.memory_dict[agent_id] = memory_matrix[index, :]

    def retrieve_memories(self, agent_ids: List[GlobalAgentId]) -> np.ndarray:
        memory_matrix = np.zeros((len(agent_ids), self.m_size), dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.memory_dict:
                memory_matrix[index, :] = self.memory_dict[agent_id]
        return memory_matrix

    def retrieve_previous_memories(self, agent_ids: List[GlobalAgentId]) -> np.ndarray:
        memory_matrix = np.zeros((len(agent_ids), self.m_size), dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.previous_memory_dict:
                memory_matrix[index, :] = self.previous_memory_dict[agent_id]
        return memory_matrix

    def remove_memories(self, agent_ids: List[GlobalAgentId]) -> None:
        for agent_id in agent_ids:
            if agent_id in self.memory_dict:
                self.memory_dict.pop(agent_id)
            if agent_id in self.previous_memory_dict:
                self.previous_memory_dict.pop(agent_id)

    def make_empty_previous_action(self, num_agents: int) -> np.ndarray:
        """
        Creates empty previous action for use with RNNs and discrete control
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros(
            (num_agents, self.behavior_spec.action_spec.discrete_size), dtype=np.int32
        )

    def save_previous_action(
        self, agent_ids: List[GlobalAgentId], action_tuple: ActionTuple
    ) -> None:
        for index, agent_id in enumerate(agent_ids):
            self.previous_action_dict[agent_id] = action_tuple.discrete[index, :]

    def retrieve_previous_action(self, agent_ids: List[GlobalAgentId]) -> np.ndarray:
        action_matrix = self.make_empty_previous_action(len(agent_ids))
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.previous_action_dict:
                action_matrix[index, :] = self.previous_action_dict[agent_id]
        return action_matrix

    def remove_previous_action(self, agent_ids: List[GlobalAgentId]) -> None:
        for agent_id in agent_ids:
            if agent_id in self.previous_action_dict:
                self.previous_action_dict.pop(agent_id)

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        raise NotImplementedError

    @staticmethod
    def check_nan_action(action: Optional[ActionTuple]) -> None:
        # Fast NaN check on the action
        # See https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy for background.
        if action is not None:
            d = np.sum(action.continuous)
            has_nan = np.isnan(d)
            if has_nan:
                raise RuntimeError("Continuous NaN action detected.")

    @abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        pass

    @abstractmethod
    def increment_step(self, n_steps):
        pass

    @abstractmethod
    def get_current_step(self):
        pass

    @abstractmethod
    def load_weights(self, values: List[np.ndarray]) -> None:
        pass

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        return []

    @abstractmethod
    def init_load_weights(self) -> None:
        pass

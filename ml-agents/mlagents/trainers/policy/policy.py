from abc import abstractmethod
from typing import Dict, List, Optional
import numpy as np

from mlagents_envs.base_env import DecisionSteps
from mlagents_envs.exception import UnityException

from mlagents.trainers.action_info import ActionInfo
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.settings import TrainerSettings, NetworkSettings


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
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
    ):
        self.behavior_spec = behavior_spec
        self.trainer_settings = trainer_settings
        self.network_settings: NetworkSettings = trainer_settings.network_settings
        self.seed = seed
        if (
            self.behavior_spec.action_spec.continuous_size > 0
            and self.behavior_spec.action_spec.discrete_size > 0
        ):
            raise UnityPolicyException("Trainers do not support mixed action spaces.")
        self.act_size = (
            list(self.behavior_spec.action_spec.discrete_branches)
            if self.behavior_spec.action_spec.is_discrete()
            else [self.behavior_spec.action_spec.continuous_size]
        )
        self.vec_obs_size = sum(
            shape[0] for shape in behavior_spec.observation_shapes if len(shape) == 1
        )
        self.vis_obs_size = sum(
            1 for shape in behavior_spec.observation_shapes if len(shape) == 3
        )
        self.use_continuous_act = self.behavior_spec.action_spec.is_continuous()
        # This line will be removed in the ActionBuffer change
        self.num_branches = (
            self.behavior_spec.action_spec.continuous_size
            + self.behavior_spec.action_spec.discrete_size
        )
        self.previous_action_dict: Dict[str, np.array] = {}
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
        self.reparameterize = reparameterize
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

    @staticmethod
    def check_nan_action(action: Optional[np.ndarray]) -> None:
        # Fast NaN check on the action
        # See https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy for background.
        if action is not None:
            d = np.sum(action)
            has_nan = np.isnan(d)
            if has_nan:
                raise RuntimeError("NaN action detected.")

    @abstractmethod
    def update_normalization(self, vector_obs: np.ndarray) -> None:
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

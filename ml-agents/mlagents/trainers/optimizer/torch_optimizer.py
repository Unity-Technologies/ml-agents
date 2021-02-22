from typing import Dict, Optional, Tuple, List
from mlagents.torch_utils import torch
import numpy as np
import math

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch.components.bc.module import BCModule
from mlagents.trainers.torch.components.reward_providers import create_reward_provider

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch.utils import ModelUtils


class TorchOptimizer(Optimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__()
        self.policy = policy
        self.trainer_settings = trainer_settings
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.memory_in: torch.Tensor = None
        self.memory_out: torch.Tensor = None
        self.m_size: int = 0
        self.global_step = torch.tensor(0)
        self.bc_module: Optional[BCModule] = None
        self.create_reward_signals(trainer_settings.reward_signals)
        self.critic_memory_dict: Dict[str, torch.Tensor] = {}
        if trainer_settings.behavioral_cloning is not None:
            self.bc_module = BCModule(
                self.policy,
                trainer_settings.behavioral_cloning,
                policy_learning_rate=trainer_settings.hyperparameters.learning_rate,
                default_batch_size=trainer_settings.hyperparameters.batch_size,
                default_num_epoch=3,
            )

    @property
    def critic(self):
        raise NotImplementedError

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        pass

    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal, settings in reward_signal_configs.items():
            # Name reward signals by string in case we have duplicates later
            self.reward_signals[reward_signal.value] = create_reward_provider(
                reward_signal, self.policy.behavior_spec, settings
            )

    def _evaluate_by_sequence(
        self, tensor_obs: List[torch.Tensor], initial_memory: np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], AgentBufferField, torch.Tensor]:
        """
        Evaluate the batch sequence-by-sequence, assembling the result. This enables us to get the
        intermediate memories for the critic.
        """
        num_experiences = tensor_obs[0].shape[0]
        all_next_memories = AgentBufferField()
        # In the buffer, the 1st sequence are the ones that are padded. So if seq_len = 3 and
        # trajectory is of length 10, the 1st sequence is [pad,pad,obs].
        # Compute the number of elements in this padded seq.
        leftover = num_experiences % self.policy.sequence_length

        # Compute values for the potentially truncated initial sequence
        _mem = initial_memory
        seq_obs = []

        first_seq_len = self.policy.sequence_length
        for _obs in tensor_obs:
            if leftover > 0:
                first_seq_obs = _obs[0:leftover]
                first_seq_len = leftover
            else:
                first_seq_obs = _obs[0 : self.policy.sequence_length]
            seq_obs.append(first_seq_obs)

        for _ in range(first_seq_len):
            all_next_memories.append(initial_memory.squeeze().detach().numpy())

        init_values, _mem = self.critic.critic_pass(
            seq_obs, _mem, sequence_length=first_seq_len
        )
        all_values = {
            signal_name: [init_values[signal_name]]
            for signal_name in init_values.keys()
        }

        # Evaluate other trajectories
        for seq_num in range(
            1, math.ceil((num_experiences) / (self.policy.sequence_length))
        ):
            seq_obs = []
            for _ in range(self.policy.sequence_length):
                all_next_memories.append(_mem.squeeze().detach().numpy())
            for _obs in tensor_obs:
                start = seq_num * self.policy.sequence_length - (
                    self.policy.sequence_length - leftover
                )
                end = (seq_num + 1) * self.policy.sequence_length - (
                    self.policy.sequence_length - leftover
                )
                seq_obs.append(_obs[start:end])
            values, _mem = self.critic.critic_pass(
                seq_obs, _mem, sequence_length=self.policy.sequence_length
            )
            for signal_name, _val in values.items():
                all_values[signal_name].append(_val)
            print("Ooga")
        # Create one tensor per reward signal
        all_value_tensors = {
            signal_name: torch.cat(value_list, dim=0)
            for signal_name, value_list in all_values.items()
        }
        next_mem = _mem
        return all_value_tensors, all_next_memories, next_mem

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        n_obs = len(self.policy.behavior_spec.observation_specs)

        if agent_id in self.critic_memory_dict:
            memory = self.critic_memory_dict[agent_id]
        else:
            memory = (
                torch.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )

        # Convert to tensors
        current_obs = [
            ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)
        ]
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        # If we're using LSTM, we want to get all the intermediate memories.
        all_next_memories: Optional[AgentBufferField] = None
        if self.policy.use_recurrent:
            (
                value_estimates,
                all_next_memories,
                next_memory,
            ) = self._evaluate_by_sequence(current_obs, memory)
        else:
            value_estimates, next_memory = self.critic.critic_pass(
                current_obs, memory, sequence_length=batch.num_experiences
            )

        # Store the memory for the next trajectory
        self.critic_memory_dict[agent_id] = next_memory

        next_value_estimate, _ = self.critic.critic_pass(
            next_obs, next_memory, sequence_length=1
        )

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
            next_value_estimate[name] = ModelUtils.to_numpy(next_value_estimate[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0
            if agent_id in self.critic_memory_dict:
                self.critic_memory_dict.pop(agent_id)
        return value_estimates, next_value_estimate, all_next_memories

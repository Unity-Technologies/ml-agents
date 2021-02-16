from typing import Dict, Optional, Tuple, List
from mlagents.torch_utils import torch
import numpy as np

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

        # If we're using LSTM, we want to get all the intermediate memories.
        all_next_memories: Optional[AgentBufferField] = None
        if self.policy.use_recurrent:
            resequenced_buffer = AgentBuffer()
            all_next_memories = AgentBufferField()
            # The 1st sequence are the ones that are padded. So if seq_len = 3 and
            # trajectory is of length 10, the ist sequence is [pad,pad,obs].
            # Compute the number of elements in this padded seq.
            leftover = batch.num_experiences % self.policy.sequence_length
            first_seq_len = self.policy.sequence_length if leftover == 0 else leftover
            for _ in range(first_seq_len):
                all_next_memories.append(memory.squeeze().detach().numpy())

            batch.resequence_and_append(
                resequenced_buffer, training_length=self.policy.sequence_length
            )
            reseq_obs = ObsUtil.from_buffer(resequenced_buffer, n_obs)
            reseq_obs = [ModelUtils.list_to_tensor(obs) for obs in reseq_obs]
            # By now, the buffer should be of length seq_len * num_seq, padded
            _mem = memory
            for seq_num in range(
                resequenced_buffer.num_experiences // self.policy.sequence_length - 1
            ):
                seq_obs = []
                for _obs in reseq_obs:
                    start = seq_num * self.policy.sequence_length
                    end = (seq_num + 1) * self.policy.sequence_length
                    seq_obs.append(_obs[start:end])
                _, next_seq_mem = self.critic.critic_pass(
                    seq_obs, _mem, sequence_length=self.policy.sequence_length
                )
                for _ in range(self.policy.sequence_length):
                    all_next_memories.append(next_seq_mem.squeeze().detach().numpy())

        # Convert to tensors
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        next_obs = [obs.unsqueeze(0) for obs in next_obs]

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

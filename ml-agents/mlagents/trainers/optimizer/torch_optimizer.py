from typing import Dict, Optional, Tuple, List
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch.components.bc.module import BCModule
from mlagents.trainers.torch.components.reward_providers import create_reward_provider

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch.utils import ModelUtils


class TorchOptimizer(Optimizer):  # pylint: disable=W0223
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
        if trainer_settings.behavioral_cloning is not None:
            self.bc_module = BCModule(
                self.policy,
                trainer_settings.behavioral_cloning,
                policy_learning_rate=trainer_settings.hyperparameters.learning_rate,
                default_batch_size=trainer_settings.hyperparameters.batch_size,
                default_num_epoch=3,
            )

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
        next_critic_obs: List[List[np.ndarray]],
        done: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        n_obs = len(self.policy.behavior_spec.sensor_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)

        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        memory = torch.zeros([1, 1, self.policy.m_size])

        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        critic_obs_np = AgentBuffer.obs_list_list_to_obs_batch(batch["critic_obs"])
        critic_obs = [
            ModelUtils.list_to_tensor_list(_agent_obs) for _agent_obs in critic_obs_np
        ]
        next_critic_obs = [
            ModelUtils.list_to_tensor_list(_obs) for _obs in next_critic_obs
        ]

        memory = torch.zeros([1, 1, self.policy.m_size])

        value_estimates, next_memory = self.policy.actor_critic.critic_pass(
            current_obs, memory, sequence_length=batch.num_experiences, critic_obs=critic_obs
        )

        next_value_estimate, _ = self.policy.actor_critic.critic_pass(
            next_obs, next_memory, sequence_length=1, critic_obs=next_critic_obs
        )

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
            next_value_estimate[name] = ModelUtils.to_numpy(next_value_estimate[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0

        return value_estimates, next_value_estimate

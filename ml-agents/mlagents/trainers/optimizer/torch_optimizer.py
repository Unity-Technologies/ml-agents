from typing import Dict, Optional, Tuple, List
from mlagents.torch_utils import torch
from mlagents.trainers.torch.agent_action import AgentAction
import numpy as np

from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil, TeamObsUtil
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
        all_dones: bool,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:

        n_obs = len(self.policy.behavior_spec.sensor_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        team_obs = TeamObsUtil.from_buffer(batch, n_obs)
        #next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        #next_team_obs = TeamObsUtil.from_buffer_next(batch, n_obs)

        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        team_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _teammate_obs]
            for _teammate_obs in team_obs
        ]
        #next_team_obs = [
        #    [ModelUtils.list_to_tensor(obs) for obs in _teammate_obs]
        #    for _teammate_obs in next_team_obs
        #]

        actions = AgentAction.from_dict(batch)
        team_actions = AgentAction.from_team_dict(batch)
        #next_actions = AgentAction.from_dict_next(batch)
        #next_team_actions = AgentAction.from_team_dict_next(batch)

        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        # critic_obs = TeamObsUtil.from_buffer(batch, n_obs)
        # critic_obs = [
        #    [ModelUtils.list_to_tensor(obs) for obs in _teammate_obs]
        #    for _teammate_obs in critic_obs
        # ]
        next_critic_obs = [
            ModelUtils.list_to_tensor_list(_list_obs) for _list_obs in next_critic_obs
         ]
        # Expand dimensions of next critic obs
        next_critic_obs = [
            [_obs.unsqueeze(0) for _obs in _list_obs] for _list_obs in next_critic_obs
         ]

        memory = torch.zeros([1, 1, self.policy.m_size])

        q_estimates, baseline_estimates, mem = self.policy.actor_critic.critic_pass(
            current_obs,
            actions,
            memory,
            sequence_length=batch.num_experiences,
            team_obs=team_obs,
            team_act=team_actions,
        )

        value_estimates, mem = self.policy.actor_critic.target_critic_value(
            current_obs,
            memory,
            sequence_length=batch.num_experiences,
            team_obs=team_obs,
        )

        boot_value_estimates, mem = self.policy.actor_critic.target_critic_value(
            next_obs,
            memory,
            sequence_length=batch.num_experiences,
            team_obs=next_critic_obs,
        )

        #next_value_estimates, next_marg_val_estimates, next_mem = self.policy.actor_critic.target_critic_pass(
        #    next_obs,
        #    next_actions,
        #    memory,
        #    sequence_length=batch.num_experiences,
        #    team_obs=next_team_obs,
        #    team_act=next_team_actions,
        #)

        # # Actions is a hack here, we need the next actions
        # next_value_estimate, next_marg_val_estimate, _ = self.policy.actor_critic.critic_pass(
        #     next_obs, actions, next_memory, sequence_length=1, critic_obs=next_critic_obs
        # )
        # These aren't used in COMAttention

        for name, estimate in baseline_estimates.items():
            baseline_estimates[name] = ModelUtils.to_numpy(estimate)

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)

        # the base line and V shpuld  not be on the same done flag
        for name, estimate in boot_value_estimates.items():
            boot_value_estimates[name] = ModelUtils.to_numpy(estimate)

        if all_dones:
            for k in boot_value_estimates:
                if not self.reward_signals[k].ignore_done:
                    boot_value_estimates[k][-1] = 0.0
        #            else:
        #                print(len(next_critic_obs))
        #                print(baseline_estimates)
        #                print(value_estimates)
        #                print(boot_value_baseline[k][-1])

        return (
            value_estimates,
            baseline_estimates,
            boot_value_estimates,
        )

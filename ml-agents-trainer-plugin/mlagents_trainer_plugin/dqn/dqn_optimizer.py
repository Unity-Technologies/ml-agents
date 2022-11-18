from typing import cast
from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from typing import List, Dict, Tuple, Optional, Union, Any
from mlagents.trainers.torch_entities.networks import ValueNetwork, Actor
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from mlagents.trainers.settings import ScheduleType, NetworkSettings

from mlagents.trainers.torch_entities.networks import Critic
import numpy as np
import attr


# TODO: fix saving to onnx


@attr.s(auto_attribs=True)
class DQNSettings(OffPolicyHyperparamSettings):
    gamma: float = 0.99
    exploration_schedule: ScheduleType = ScheduleType.LINEAR
    exploration_initial_eps: float = 0.1
    exploration_final_eps: float = 0.05
    target_update_interval: int = 10000
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class DQNOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)

        # initialize hyper parameters
        params = list(self.policy.actor.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stream_names = list(self.reward_signals.keys())
        self.gammas = [_val.gamma for _val in trainer_settings.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }

        self.hyperparameters: DQNSettings = cast(
            DQNSettings, trainer_settings.hyperparameters
        )
        self.tau = self.hyperparameters.tau
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )

        self.decay_exploration_rate = ModelUtils.DecayedValue(
            self.hyperparameters.exploration_schedule,
            self.hyperparameters.exploration_initial_eps,
            self.hyperparameters.exploration_final_eps,
            20000,
        )

        # initialize Target Q_network
        self.q_net_target = QNetwork(
            stream_names=self.reward_signals.keys(),
            observation_specs=policy.behavior_spec.observation_specs,
            network_settings=policy.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        ModelUtils.soft_update(self.policy.actor, self.q_net_target, 1.0)

        self.q_net_target.to(default_device())

    @property
    def critic(self):
        return self.q_net_target

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Get decayed parameters
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        exp_rate = self.decay_exploration_rate.get_value(self.policy.get_current_step())
        self.policy.actor.exploration_rate = exp_rate
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        actions = AgentAction.from_buffer(batch)

        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        current_q_values, _ = self.policy.actor.critic_pass(
            current_obs, sequence_length=self.policy.sequence_length
        )

        qloss = []
        with torch.no_grad():
            greedy_actions = self.policy.actor.get_greedy_action(current_q_values)
            next_q_values_list, _ = self.q_net_target.critic_pass(
                next_obs, sequence_length=self.policy.sequence_length
            )
        for name_i, name in enumerate(rewards.keys()):
            with torch.no_grad():
                next_q_values = torch.gather(
                    next_q_values_list[name], dim=1, index=greedy_actions
                ).squeeze()
                target_q_values = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[name_i]
                    * next_q_values
                )
                target_q_values = target_q_values.reshape(-1, 1)
            curr_q = torch.gather(
                current_q_values[name], dim=1, index=actions.discrete_tensor
            )
            qloss.append(torch.nn.functional.smooth_l1_loss(curr_q, target_q_values))

        loss = torch.mean(torch.stack(qloss))
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ModelUtils.soft_update(self.policy.actor, self.q_net_target, self.tau)
        update_stats = {
            "Losses/Value Loss": loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/epsilon": exp_rate,
        }

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))
        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self.critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules


class QNetwork(nn.Module, Actor, Critic):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        exploration_initial_eps: float = 1.0,
    ):
        self.exploration_rate = exploration_initial_eps
        nn.Module.__init__(self)
        output_act_size = max(sum(action_spec.discrete_branches), 1)
        self.network_body = ValueNetwork(
            stream_names,
            observation_specs,
            network_settings,
            outputs_per_stream=output_act_size,
        )

        # extra tensors for exporting to ONNX
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.act_size_vector_deprecated = torch.nn.Parameter(
            torch.Tensor(
                [
                    self.action_spec.continuous_size
                    + sum(self.action_spec.discrete_branches)
                ]
            ),
            requires_grad=False,
        )
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        value_outputs, critic_mem_out = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        return value_outputs, critic_mem_out

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        out_vals, memories = self.critic_pass(inputs, memories, sequence_length)

        # fixme random action tensor
        export_out = [self.version_number, self.memory_size_vector]

        disc_action_out = self.get_greedy_action(out_vals)
        deterministic_disc_action_out = self.get_random_action(out_vals)
        export_out += [
            disc_action_out,
            self.discrete_act_size_vector,
            deterministic_disc_action_out,
        ]
        return tuple(export_out)

    def get_random_action(self, inputs) -> torch.Tensor:
        action_out = torch.randint(
            0, self.action_spec.discrete_branches[0], (len(inputs), 1)
        )
        return action_out

    @staticmethod
    def get_greedy_action(q_values) -> torch.Tensor:
        all_q = torch.cat([val.unsqueeze(0) for val in q_values.values()])
        return torch.argmax(all_q.sum(dim=0), dim=1, keepdim=True)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        deterministic=False,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        run_out = {}
        if not deterministic and np.random.rand() < self.exploration_rate:
            action_out = self.get_random_action(inputs)
            action_out = AgentAction(None, [action_out])
            run_out["env_action"] = action_out.to_action_tuple()
        else:
            out_vals, _ = self.critic_pass(inputs, memories, sequence_length)
            action_out = self.get_greedy_action(out_vals)
            action_out = AgentAction(None, [action_out])
            run_out["env_action"] = action_out.to_action_tuple()
        return action_out, run_out, torch.Tensor([])

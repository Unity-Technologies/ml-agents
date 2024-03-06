from typing import Dict, cast, List, Tuple, Optional
from collections import defaultdict
import attr

from mlagents.trainers.torch_entities.components.reward_providers.extrinsic_reward_provider import (
    ExtrinsicRewardProvider,
)
import numpy as np
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import (
    AgentBuffer,
    BufferKey,
    RewardSignalUtil,
    AgentBufferField,
)

from mlagents_envs.timers import timed
from mlagents_envs.base_env import ObservationSpec, ActionSpec
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    RewardSignalSettings,
    RewardSignalType,
    TrainerSettings,
    NetworkSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import Critic, MultiAgentNetworkBody
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil, GroupObsUtil

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class POCASettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR


class TorchPOCAOptimizer(TorchOptimizer):
    class POCAValueNetwork(torch.nn.Module, Critic):
        """
        The POCAValueNetwork uses the MultiAgentNetworkBody to compute the value
        and POCA baseline for a variable number of agents in a group that all
        share the same observation and action space.
        """

        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            torch.nn.Module.__init__(self)
            self.network_body = MultiAgentNetworkBody(
                observation_specs, network_settings, action_spec
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units

            self.value_heads = ValueHeads(stream_names, encoding_size + 1, 1)
            # The + 1 is for the normalized number of agents

        @property
        def memory_size(self) -> int:
            return self.network_body.memory_size

        def update_normalization(self, buffer: AgentBuffer) -> None:
            self.network_body.update_normalization(buffer)

        def baseline(
            self,
            obs_without_actions: List[torch.Tensor],
            obs_with_actions: Tuple[List[List[torch.Tensor]], List[AgentAction]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            """
            The POCA baseline marginalizes the action of the agent associated with self_obs.
            It calls the forward pass of the MultiAgentNetworkBody with the state action
            pairs of groupmates but just the state of the agent in question.
            :param obs_without_actions: The obs of the agent for which to compute the baseline.
            :param obs_with_actions: Tuple of observations and actions for all groupmates.
            :param memories: If using memory, a Tensor of initial memories.
            :param sequence_length: If using memory, the sequence length.

            :return: A Tuple of Dict of reward stream to tensor and critic memories.
            """
            (obs, actions) = obs_with_actions
            encoding, memories = self.network_body(
                obs_only=[obs_without_actions],
                obs=obs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

        def critic_pass(
            self,
            obs: List[List[torch.Tensor]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            """
            A centralized value function. It calls the forward pass of MultiAgentNetworkBody
            with just the states of all agents.
            :param obs: List of observations for all agents in group
            :param memories: If using memory, a Tensor of initial memories.
            :param sequence_length: If using memory, the sequence length.
            :return: A Tuple of Dict of reward stream to tensor and critic memories.
            """
            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

        def forward(
            self,
            encoding: torch.Tensor,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            output = self.value_heads(encoding)
            return output, memories

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        :param policy: A TorchPolicy object that will be updated by this POCA Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self._critic = TorchPOCAOptimizer.POCAValueNetwork(
            reward_signal_names,
            policy.behavior_spec.observation_specs,
            network_settings=trainer_settings.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        # Move to GPU if needed
        self._critic.to(default_device())

        params = list(self.policy.actor.parameters()) + list(self.critic.parameters())

        self.hyperparameters: POCASettings = cast(
            POCASettings, trainer_settings.hyperparameters
        )

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        self.value_memory_dict: Dict[str, torch.Tensor] = {}
        self.baseline_memory_dict: Dict[str, torch.Tensor] = {}

    def create_reward_signals(
        self, reward_signal_configs: Dict[RewardSignalType, RewardSignalSettings]
    ) -> None:
        """
        Create reward signals. Override default to provide warnings for Curiosity and
        GAIL, and make sure Extrinsic adds team rewards.
        :param reward_signal_configs: Reward signal config.
        """
        for reward_signal in reward_signal_configs.keys():
            if reward_signal != RewardSignalType.EXTRINSIC:
                logger.warning(
                    f"Reward signal {reward_signal.value.capitalize()} is not supported with the POCA trainer; "
                    "results may be unexpected."
                )
        super().create_reward_signals(reward_signal_configs)
        # Make sure we add the groupmate rewards in POCA, so agents learn how to help each
        # other achieve individual rewards as well
        for reward_provider in self.reward_signals.values():
            if isinstance(reward_provider, ExtrinsicRewardProvider):
                reward_provider.add_groupmate_rewards = True

    @property
    def critic(self):
        return self._critic

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
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        returns = {}
        old_values = {}
        old_baseline_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )
            old_baseline_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.baseline_estimates_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs]
            for _groupmate_obs in groupmate_obs
        ]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)
        groupmate_actions = AgentAction.group_from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]

        baseline_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.BASELINE_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.BASELINE_MEMORY]), self.policy.sequence_length
            )
        ]

        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)
            baseline_memories = torch.stack(baseline_memories).unsqueeze(0)

        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]
        entropy = run_out["entropy"]

        all_obs = [current_obs] + groupmate_obs
        values, _ = self.critic.critic_pass(
            all_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
        baselines, _ = self.critic.baseline(
            current_obs,
            groupmate_obs_and_actions,
            memories=baseline_memories,
            sequence_length=self.policy.sequence_length,
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        baseline_loss = ModelUtils.trust_region_value_loss(
            baselines, old_baseline_values, returns, decay_eps, loss_masks
        )
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )

        loss = (
            policy_loss
            + 0.5 * (value_loss + 0.5 * baseline_loss)
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Baseline Loss": baseline_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        return update_stats

    def get_modules(self):
        modules = {"Optimizer:adam": self.optimizer, "Optimizer:critic": self._critic}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    def _evaluate_by_sequence_team(
        self,
        self_obs: List[torch.Tensor],
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        init_value_mem: torch.Tensor,
        init_baseline_mem: torch.Tensor,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        AgentBufferField,
        AgentBufferField,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Evaluate a trajectory sequence-by-sequence, assembling the result. This enables us to get the
        intermediate memories for the critic.
        :param tensor_obs: A List of tensors of shape (trajectory_len, <obs_dim>) that are the agent's
            observations for this trajectory.
        :param initial_memory: The memory that preceeds this trajectory. Of shape (1,1,<mem_size>), i.e.
            what is returned as the output of a MemoryModules.
        :return: A Tuple of the value estimates as a Dict of [name, tensor], an AgentBufferField of the initial
            memories to be used during value function update, and the final memory at the end of the trajectory.
        """
        num_experiences = self_obs[0].shape[0]
        all_next_value_mem = AgentBufferField()
        all_next_baseline_mem = AgentBufferField()

        # When using LSTM, we need to divide the trajectory into sequences of equal length. Sometimes,
        # that division isn't even, and we must pad the leftover sequence.
        # In the buffer, the last sequence are the ones that are padded. So if seq_len = 3 and
        # trajectory is of length 10, the last sequence is [obs,pad,pad].
        # Compute the number of elements in this padded seq.
        leftover_seq_len = num_experiences % self.policy.sequence_length

        all_values: Dict[str, List[np.ndarray]] = defaultdict(list)
        all_baseline: Dict[str, List[np.ndarray]] = defaultdict(list)
        _baseline_mem = init_baseline_mem
        _value_mem = init_value_mem

        # Evaluate other trajectories, carrying over _mem after each
        # trajectory
        for seq_num in range(num_experiences // self.policy.sequence_length):
            for _ in range(self.policy.sequence_length):
                all_next_value_mem.append(ModelUtils.to_numpy(_value_mem.squeeze()))
                all_next_baseline_mem.append(
                    ModelUtils.to_numpy(_baseline_mem.squeeze())
                )

            start = seq_num * self.policy.sequence_length
            end = (seq_num + 1) * self.policy.sequence_length

            self_seq_obs = []
            groupmate_seq_obs = []
            groupmate_seq_act = []
            seq_obs = []
            for _self_obs in self_obs:
                seq_obs.append(_self_obs[start:end])
            self_seq_obs.append(seq_obs)

            for groupmate_obs, groupmate_action in zip(obs, actions):
                seq_obs = []
                for _obs in groupmate_obs:
                    sliced_seq_obs = _obs[start:end]
                    seq_obs.append(sliced_seq_obs)
                groupmate_seq_obs.append(seq_obs)
                _act = groupmate_action.slice(start, end)
                groupmate_seq_act.append(_act)

            all_seq_obs = self_seq_obs + groupmate_seq_obs
            values, _value_mem = self.critic.critic_pass(
                all_seq_obs, _value_mem, sequence_length=self.policy.sequence_length
            )
            for signal_name, _val in values.items():
                all_values[signal_name].append(_val)

            groupmate_obs_and_actions = (groupmate_seq_obs, groupmate_seq_act)
            baselines, _baseline_mem = self.critic.baseline(
                self_seq_obs[0],
                groupmate_obs_and_actions,
                _baseline_mem,
                sequence_length=self.policy.sequence_length,
            )
            for signal_name, _val in baselines.items():
                all_baseline[signal_name].append(_val)

        # Compute values for the potentially truncated initial sequence
        if leftover_seq_len > 0:
            self_seq_obs = []
            groupmate_seq_obs = []
            groupmate_seq_act = []
            seq_obs = []
            for _self_obs in self_obs:
                last_seq_obs = _self_obs[-leftover_seq_len:]
                seq_obs.append(last_seq_obs)
            self_seq_obs.append(seq_obs)

            for groupmate_obs, groupmate_action in zip(obs, actions):
                seq_obs = []
                for _obs in groupmate_obs:
                    last_seq_obs = _obs[-leftover_seq_len:]
                    seq_obs.append(last_seq_obs)
                groupmate_seq_obs.append(seq_obs)
                _act = groupmate_action.slice(len(_obs) - leftover_seq_len, len(_obs))
                groupmate_seq_act.append(_act)

            # For the last sequence, the initial memory should be the one at the
            # beginning of this trajectory.
            seq_obs = []
            for _ in range(leftover_seq_len):
                all_next_value_mem.append(ModelUtils.to_numpy(_value_mem.squeeze()))
                all_next_baseline_mem.append(
                    ModelUtils.to_numpy(_baseline_mem.squeeze())
                )

            all_seq_obs = self_seq_obs + groupmate_seq_obs
            last_values, _value_mem = self.critic.critic_pass(
                all_seq_obs, _value_mem, sequence_length=leftover_seq_len
            )
            for signal_name, _val in last_values.items():
                all_values[signal_name].append(_val)
            groupmate_obs_and_actions = (groupmate_seq_obs, groupmate_seq_act)
            last_baseline, _baseline_mem = self.critic.baseline(
                self_seq_obs[0],
                groupmate_obs_and_actions,
                _baseline_mem,
                sequence_length=leftover_seq_len,
            )
            for signal_name, _val in last_baseline.items():
                all_baseline[signal_name].append(_val)
        # Create one tensor per reward signal
        all_value_tensors = {
            signal_name: torch.cat(value_list, dim=0)
            for signal_name, value_list in all_values.items()
        }
        all_baseline_tensors = {
            signal_name: torch.cat(baseline_list, dim=0)
            for signal_name, baseline_list in all_baseline.items()
        }
        next_value_mem = _value_mem
        next_baseline_mem = _baseline_mem
        return (
            all_value_tensors,
            all_baseline_tensors,
            all_next_value_mem,
            all_next_baseline_mem,
            next_value_mem,
            next_baseline_mem,
        )

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        """
        Override base class method. Unused in the trainer, but needed to make sure class heirarchy is maintained.
        Assume that there are no group obs.
        """
        (
            value_estimates,
            _,
            next_value_estimates,
            all_next_value_mem,
            _,
        ) = self.get_trajectory_and_baseline_value_estimates(
            batch, next_obs, [], done, agent_id
        )

        return value_estimates, next_value_estimates, all_next_value_mem

    def get_trajectory_and_baseline_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        next_groupmate_obs: List[List[np.ndarray]],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, float],
        Optional[AgentBufferField],
        Optional[AgentBufferField],
    ]:
        """
        Get value estimates, baseline estimates, and memories for a trajectory, in batch form.
        :param batch: An AgentBuffer that consists of a trajectory.
        :param next_obs: the next observation (after the trajectory). Used for bootstrapping
            if this is not a terminal trajectory.
        :param next_groupmate_obs: the next observations from other members of the group.
        :param done: Set true if this is a terminal trajectory.
        :param agent_id: Agent ID of the agent that this trajectory belongs to.
        :returns: A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
            the baseline estimates as a Dict, the final value estimate as a Dict of [name, float], and
            optionally (if using memories) an AgentBufferField of initial critic and baseline memories to be used
            during update.
        """

        n_obs = len(self.policy.behavior_spec.observation_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        groupmate_obs = GroupObsUtil.from_buffer(batch, n_obs)

        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        groupmate_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _groupmate_obs]
            for _groupmate_obs in groupmate_obs
        ]

        groupmate_actions = AgentAction.group_from_buffer(batch)

        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_obs = [obs.unsqueeze(0) for obs in next_obs]

        next_groupmate_obs = [
            ModelUtils.list_to_tensor_list(_list_obs)
            for _list_obs in next_groupmate_obs
        ]
        # Expand dimensions of next critic obs
        next_groupmate_obs = [
            [_obs.unsqueeze(0) for _obs in _list_obs]
            for _list_obs in next_groupmate_obs
        ]

        if agent_id in self.value_memory_dict:
            # The agent_id should always be in both since they are added together
            _init_value_mem = self.value_memory_dict[agent_id]
            _init_baseline_mem = self.baseline_memory_dict[agent_id]
        else:
            _init_value_mem = (
                torch.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )
            _init_baseline_mem = (
                torch.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )

        all_obs = (
            [current_obs] + groupmate_obs
            if groupmate_obs is not None
            else [current_obs]
        )
        all_next_value_mem: Optional[AgentBufferField] = None
        all_next_baseline_mem: Optional[AgentBufferField] = None
        with torch.no_grad():
            if self.policy.use_recurrent:
                (
                    value_estimates,
                    baseline_estimates,
                    all_next_value_mem,
                    all_next_baseline_mem,
                    next_value_mem,
                    next_baseline_mem,
                ) = self._evaluate_by_sequence_team(
                    current_obs,
                    groupmate_obs,
                    groupmate_actions,
                    _init_value_mem,
                    _init_baseline_mem,
                )
            else:
                value_estimates, next_value_mem = self.critic.critic_pass(
                    all_obs, _init_value_mem, sequence_length=batch.num_experiences
                )
                groupmate_obs_and_actions = (groupmate_obs, groupmate_actions)
                baseline_estimates, next_baseline_mem = self.critic.baseline(
                    current_obs,
                    groupmate_obs_and_actions,
                    _init_baseline_mem,
                    sequence_length=batch.num_experiences,
                )
        # Store the memory for the next trajectory
        self.value_memory_dict[agent_id] = next_value_mem
        self.baseline_memory_dict[agent_id] = next_baseline_mem

        all_next_obs = (
            [next_obs] + next_groupmate_obs
            if next_groupmate_obs is not None
            else [next_obs]
        )

        next_value_estimates, _ = self.critic.critic_pass(
            all_next_obs, next_value_mem, sequence_length=1
        )

        for name, estimate in baseline_estimates.items():
            baseline_estimates[name] = ModelUtils.to_numpy(estimate)

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)

        # the base line and V shpuld  not be on the same done flag
        for name, estimate in next_value_estimates.items():
            next_value_estimates[name] = ModelUtils.to_numpy(estimate)

        if done:
            for k in next_value_estimates:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimates[k][-1] = 0.0

        return (
            value_estimates,
            baseline_estimates,
            next_value_estimates,
            all_next_value_mem,
            all_next_baseline_mem,
        )

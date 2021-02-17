from typing import Dict, cast
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import TrainerSettings, PPOSettings
from mlagents.trainers.torch.networks import Critic
from mlagents.trainers.torch.layers import EntityEmbedding, ResidualSelfAttention, LinearEncoder
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


class TorchCOMAOptimizer(TorchOptimizer):
    class COMAValueNetwork(torch.nn.Module, Critic):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            super().__init__()
            self.normalize = network_settings.normalize
            self.use_lstm = network_settings.memory is not None
            # Scale network depending on num agents
            self.h_size = network_settings.hidden_units
            self.m_size = (
                network_settings.memory.memory_size
                if network_settings.memory is not None
                else 0
            )
            self.processors, _input_size = ModelUtils.create_input_processors(
                sensor_specs,
                self.h_size,
                network_settings.vis_encode_type,
                normalize=self.normalize,
            )
            self.action_spec = action_spec

            # Modules for self-attention
            obs_only_ent_size = sum(_input_size)
            q_ent_size = (
                sum(_input_size)
                + sum(self.action_spec.discrete_branches)
                + self.action_spec.continuous_size
            )
            self.obs_encoder = EntityEmbedding(
                0, obs_only_ent_size, None, self.h_size, concat_self=False
            )
            self.obs_action_encoder = EntityEmbedding(
                0, q_ent_size, None, self.h_size, concat_self=False
            )

            self.self_attn = ResidualSelfAttention(self.h_size)

            self.linear_encoder = LinearEncoder(
                self.h_size,
                network_settings.num_layers,
                self.h_size,
                kernel_gain=(0.125 / self.h_size) ** 0.5,
            )

            if self.use_lstm:
                self.lstm = LSTM(self.h_size, self.m_size)
            else:
                self.lstm = None  # type: ignorek


    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, len(self.processors))
        for vec_input, enc in zip(obs, self.processors):
            if isinstance(enc, VectorInput):
                enc.update_normalization(torch.as_tensor(vec_input))

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        if self.normalize:
            for n1, n2 in zip(self.processors, other_network.processors):
                if isinstance(n1, VectorInput) and isinstance(n2, VectorInput):
                    n1.copy_normalization(n2)

    def _get_masks_from_nans(self, obs_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Get attention masks by grabbing an arbitrary obs across all the agents
        Since these are raw obs, the padded values are still NaN
        """
        only_first_obs = [_all_obs[0] for _all_obs in obs_tensors]
        obs_for_mask = torch.stack(only_first_obs, dim=1)
        # Get the mask from nans
        attn_mask = torch.any(obs_for_mask.isnan(), dim=2).type(torch.FloatTensor)
        return attn_mask

    def baseline(
        self,
        self_obs: List[List[torch.Tensor]],
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self_attn_masks = []

        f_inp = None
        concat_f_inp = []
        for inputs, action in zip(obs, actions):
            encodes = []
            for idx, processor in enumerate(self.processors):
                obs_input = inputs[idx]
                obs_input[obs_input.isnan()] = 0.0  # Remove NaNs
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            cat_encodes = [
                torch.cat(encodes, dim=-1),
                action.to_flat(self.action_spec.discrete_branches),
            ]
            concat_f_inp.append(torch.cat(cat_encodes, dim=1))

        if concat_f_inp:
            f_inp = torch.stack(concat_f_inp, dim=1)
            self_attn_masks.append(self._get_masks_from_nans(obs))

        concat_encoded_obs = []
        encodes = []
        for idx, processor in enumerate(self.processors):
            obs_input = self_obs[idx]
            obs_input[obs_input.isnan()] = 0.0  # Remove NaNs
            processed_obs = processor(obs_input)
            encodes.append(processed_obs)
        concat_encoded_obs.append(torch.cat(encodes, dim=-1))
        g_inp = torch.stack(concat_encoded_obs, dim=1)
        # Get the mask from nans
        self_attn_masks.append(self._get_masks_from_nans([self_obs]))
        encoding, memories = self.forward(
            f_inp,
            g_inp,
            self_attn_masks,
            memories=memories,
            sequence_length=sequence_length,
        )
        return encoding, memories

    def critic_pass(
        self,
        obs: List[List[torch.Tensor]],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self_attn_masks = []
        concat_encoded_obs = []
        for inputs in obs:
            encodes = []
            for idx, processor in enumerate(self.processors):
                obs_input = inputs[idx]
                obs_input[obs_input.isnan()] = 0.0  # Remove NaNs
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            concat_encoded_obs.append(torch.cat(encodes, dim=-1))
        g_inp = torch.stack(concat_encoded_obs, dim=1)
        # Get the mask from nans
        self_attn_masks.append(self._get_masks_from_nans(obs))
        encoding, memories = self.forward(
            None,
            g_inp,
            self_attn_masks,
            memories=memories,
            sequence_length=sequence_length,
        )
        return encoding, memories


    def forward(
        self,
        f_enc: torch.Tensor,
        g_enc: torch.Tensor,
        self_attn_masks: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self_attn_inputs = []

        if f_enc is not None:
            self_attn_inputs.append(self.obs_action_encoder(None, f_enc))
        if g_enc is not None:
            self_attn_inputs.append(self.obs_encoder(None, g_enc))

        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity, self_attn_masks)

        inputs = encoded_state
        encoding = self.linear_encoder(inputs)

        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        return encoding, memories

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        # Create the graph here to give more granular control of the TF graph to the Optimizer.

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        if policy.shared_critic:
            self.value_net = policy.actor
        else:
            self.value_net = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )

        params = list(self.policy.actor.parameters()) + list(
            self.value_net.parameters()
        )
        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )
        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
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

    @property
    def critic(self):
        return self.value_net

    def ppo_value_loss(
        self,
        values: Dict[str, torch.Tensor],
        old_values: Dict[str, torch.Tensor],
        returns: Dict[str, torch.Tensor],
        epsilon: float,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluates value loss for PPO.
        :param values: Value output of the current network.
        :param old_values: Value stored with experiences in buffer.
        :param returns: Computed returns.
        :param epsilon: Clipping value for value estimate.
        :param loss_mask: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        value_losses = []
        for name, head in values.items():
            old_val_tensor = old_values[name]
            returns_tensor = returns[name]
            clipped_value_estimate = old_val_tensor + torch.clamp(
                head - old_val_tensor, -1 * epsilon, epsilon
            )
            v_opt_a = (returns_tensor - head) ** 2
            v_opt_b = (returns_tensor - clipped_value_estimate) ** 2
            value_loss = ModelUtils.masked_mean(torch.max(v_opt_a, v_opt_b), loss_masks)
            value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        return value_loss

    def ppo_policy_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate PPO policy loss.
        :param advantages: Computed advantages.
        :param log_probs: Current policy probabilities
        :param old_log_probs: Past policy probabilities
        :param loss_masks: Mask for losses. Used with LSTM to ignore 0'ed out experiences.
        """
        advantage = advantages.unsqueeze(-1)

        decay_epsilon = self.hyperparameters.epsilon
        r_theta = torch.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = (
            torch.clamp(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * advantage
        )
        policy_loss = -1 * ModelUtils.masked_mean(
            torch.min(p_opt_a, p_opt_b), loss_masks
        )
        return policy_loss

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
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        log_probs, entropy = self.policy.evaluate_actions(
            current_obs,
            masks=act_masks,
            actions=actions,
            memories=memories,
            seq_len=self.policy.sequence_length,
        )
        values, _ = self.critic.critic_pass(
            current_obs, memories=memories, sequence_length=self.policy.sequence_length
        )
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        value_loss = self.ppo_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = self.ppo_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
        )
        loss = (
            policy_loss
            + 0.5 * value_loss
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
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))

        return update_stats

    def get_modules(self):
        modules = {"Optimizer": self.optimizer}
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

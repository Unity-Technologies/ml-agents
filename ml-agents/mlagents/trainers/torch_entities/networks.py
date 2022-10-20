from typing import Callable, List, Dict, Tuple, Optional, Union, Any
import abc

from mlagents.torch_utils import torch, nn

from mlagents_envs.base_env import ActionSpec, ObservationSpec, ObservationType
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.settings import NetworkSettings, EncoderType, ConditioningType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.layers import LSTM, LinearEncoder
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.conditioning import ConditionalEncoder
from mlagents.trainers.torch_entities.attention import (
    EntityEmbedding,
    ResidualSelfAttention,
    get_zero_entities_mask,
)
from mlagents.trainers.exception import UnityTrainerException


ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool], torch.Tensor
]

EPSILON = 1e-7


class ObservationEncoder(nn.Module):
    ATTENTION_EMBEDDING_SIZE = 128  # The embedding size of attention is fixed

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        h_size: int,
        vis_encode_type: EncoderType,
        normalize: bool = False,
    ):
        """
        Returns an ObservationEncoder that can process and encode a set of observations.
        Will use an RSA if needed for variable length observations.
        """
        super().__init__()
        self.processors, self.embedding_sizes = ModelUtils.create_input_processors(
            observation_specs,
            h_size,
            vis_encode_type,
            self.ATTENTION_EMBEDDING_SIZE,
            normalize=normalize,
        )
        self.rsa, self.x_self_encoder = ModelUtils.create_residual_self_attention(
            self.processors, self.embedding_sizes, self.ATTENTION_EMBEDDING_SIZE
        )
        if self.rsa is not None:
            total_enc_size = sum(self.embedding_sizes) + self.ATTENTION_EMBEDDING_SIZE
        else:
            total_enc_size = sum(self.embedding_sizes)
        self.normalize = normalize
        self._total_enc_size = total_enc_size

        self._total_goal_enc_size = 0
        self._goal_processor_indices: List[int] = []
        for i in range(len(observation_specs)):
            if observation_specs[i].observation_type == ObservationType.GOAL_SIGNAL:
                self._total_goal_enc_size += self.embedding_sizes[i]
                self._goal_processor_indices.append(i)

    @property
    def total_enc_size(self) -> int:
        """
        Returns the total encoding size for this ObservationEncoder.
        """
        return self._total_enc_size

    @property
    def total_goal_enc_size(self) -> int:
        """
        Returns the total goal encoding size for this ObservationEncoder.
        """
        return self._total_goal_enc_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        obs = ObsUtil.from_buffer(buffer, len(self.processors))
        for vec_input, enc in zip(obs, self.processors):
            if isinstance(enc, VectorInput):
                enc.update_normalization(torch.as_tensor(vec_input.to_ndarray()))

    def copy_normalization(self, other_encoder: "ObservationEncoder") -> None:
        if self.normalize:
            for n1, n2 in zip(self.processors, other_encoder.processors):
                if isinstance(n1, VectorInput) and isinstance(n2, VectorInput):
                    n1.copy_normalization(n2)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations using a list of processors and an RSA.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        encodes = []
        var_len_processor_inputs: List[Tuple[nn.Module, torch.Tensor]] = []

        for idx, processor in enumerate(self.processors):
            if not isinstance(processor, EntityEmbedding):
                # The input can be encoded without having to process other inputs
                obs_input = inputs[idx]
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            else:
                var_len_processor_inputs.append((processor, inputs[idx]))
        if len(encodes) != 0:
            encoded_self = torch.cat(encodes, dim=1)
            input_exist = True
        else:
            input_exist = False
        if len(var_len_processor_inputs) > 0 and self.rsa is not None:
            # Some inputs need to be processed with a variable length encoder
            masks = get_zero_entities_mask([p_i[1] for p_i in var_len_processor_inputs])
            embeddings: List[torch.Tensor] = []
            processed_self = (
                self.x_self_encoder(encoded_self)
                if input_exist and self.x_self_encoder is not None
                else None
            )
            for processor, var_len_input in var_len_processor_inputs:
                embeddings.append(processor(processed_self, var_len_input))
            qkv = torch.cat(embeddings, dim=1)
            attention_embedding = self.rsa(qkv, masks)
            if not input_exist:
                encoded_self = torch.cat([attention_embedding], dim=1)
                input_exist = True
            else:
                encoded_self = torch.cat([encoded_self, attention_embedding], dim=1)

        if not input_exist:
            raise UnityTrainerException(
                "The trainer was unable to process any of the provided inputs. "
                "Make sure the trained agents has at least one sensor attached to them."
            )

        return encoded_self

    def get_goal_encoding(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode observations corresponding to goals using a list of processors.
        :param inputs: List of Tensors corresponding to a set of obs.
        """
        encodes = []
        for idx in self._goal_processor_indices:
            processor = self.processors[idx]
            if not isinstance(processor, EntityEmbedding):
                # The input can be encoded without having to process other inputs
                obs_input = inputs[idx]
                processed_obs = processor(obs_input)
                encodes.append(processed_obs)
            else:
                raise UnityTrainerException(
                    "The one of the goals uses variable length observations. This use "
                    "case is not supported."
                )
        if len(encodes) != 0:
            encoded = torch.cat(encodes, dim=1)
        else:
            raise UnityTrainerException(
                "Trainer was unable to process any of the goals provided as input."
            )
        return encoded


class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        self.processors = self.observation_encoder.processors
        total_enc_size = self.observation_encoder.total_enc_size
        total_enc_size += encoded_act_size

        if (
            self.observation_encoder.total_goal_enc_size > 0
            and network_settings.goal_conditioning_type == ConditioningType.HYPER
        ):
            self._body_endoder = ConditionalEncoder(
                total_enc_size,
                self.observation_encoder.total_goal_enc_size,
                self.h_size,
                network_settings.num_layers,
                1,
            )
        else:
            self._body_endoder = LinearEncoder(
                total_enc_size, network_settings.num_layers, self.h_size
            )

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_self = self.observation_encoder(inputs)
        if actions is not None:
            encoded_self = torch.cat([encoded_self, actions], dim=1)
        if isinstance(self._body_endoder, ConditionalEncoder):
            goal = self.observation_encoder.get_goal_encoding(inputs)
            encoding = self._body_endoder(encoded_self, goal)
        else:
            encoding = self._body_endoder(encoded_self)

        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        return encoding, memories


class MultiAgentNetworkBody(torch.nn.Module):
    """
    A network body that uses a self attention layer to handle state
    and action input from a potentially variable number of agents that
    share the same observation and action space.
    """

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
    ):
        super().__init__()
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )
        self.action_spec = action_spec
        self.observation_encoder = ObservationEncoder(
            observation_specs,
            self.h_size,
            network_settings.vis_encode_type,
            self.normalize,
        )
        self.processors = self.observation_encoder.processors

        # Modules for multi-agent self-attention
        obs_only_ent_size = self.observation_encoder.total_enc_size
        q_ent_size = (
            obs_only_ent_size
            + sum(self.action_spec.discrete_branches)
            + self.action_spec.continuous_size
        )

        attention_embeding_size = self.h_size
        self.obs_encoder = EntityEmbedding(
            obs_only_ent_size, None, attention_embeding_size
        )
        self.obs_action_encoder = EntityEmbedding(
            q_ent_size, None, attention_embeding_size
        )

        self.self_attn = ResidualSelfAttention(attention_embeding_size)

        self.linear_encoder = LinearEncoder(
            attention_embeding_size,
            network_settings.num_layers,
            self.h_size,
            kernel_gain=(0.125 / self.h_size) ** 0.5,
        )

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore
        self._current_max_agents = torch.nn.Parameter(
            torch.as_tensor(1), requires_grad=False
        )

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.observation_encoder.update_normalization(buffer)

    def copy_normalization(self, other_network: "MultiAgentNetworkBody") -> None:
        self.observation_encoder.copy_normalization(other_network.observation_encoder)

    def _get_masks_from_nans(self, obs_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Get attention masks by grabbing an arbitrary obs across all the agents
        Since these are raw obs, the padded values are still NaN
        """
        only_first_obs = [_all_obs[0] for _all_obs in obs_tensors]
        # Just get the first element in each obs regardless of its dimension. This will speed up
        # searching for NaNs.
        only_first_obs_flat = torch.stack(
            [_obs.flatten(start_dim=1)[:, 0] for _obs in only_first_obs], dim=1
        )
        # Get the mask from NaNs
        attn_mask = only_first_obs_flat.isnan().float()
        return attn_mask

    def _copy_and_remove_nans_from_obs(
        self, all_obs: List[List[torch.Tensor]], attention_mask: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """
        Helper function to remove NaNs from observations using an attention mask.
        """
        obs_with_no_nans = []
        for i_agent, single_agent_obs in enumerate(all_obs):
            no_nan_obs = []
            for obs in single_agent_obs:
                new_obs = obs.clone()
                new_obs[attention_mask.bool()[:, i_agent], ::] = 0.0  # Remove NaNs fast
                no_nan_obs.append(new_obs)
            obs_with_no_nans.append(no_nan_obs)
        return obs_with_no_nans

    def forward(
        self,
        obs_only: List[List[torch.Tensor]],
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param obs_only: Observations to be processed that do not have corresponding actions.
            These are encoded with the obs_encoder.
        :param obs: Observations to be processed that do have corresponding actions.
            After concatenation with actions, these are processed with obs_action_encoder.
        :param actions: After concatenation with obs, these are processed with obs_action_encoder.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        """
        self_attn_masks = []
        self_attn_inputs = []
        concat_f_inp = []
        if obs:
            obs_attn_mask = self._get_masks_from_nans(obs)
            obs = self._copy_and_remove_nans_from_obs(obs, obs_attn_mask)
            for inputs, action in zip(obs, actions):
                encoded = self.observation_encoder(inputs)
                cat_encodes = [
                    encoded,
                    action.to_flat(self.action_spec.discrete_branches),
                ]
                concat_f_inp.append(torch.cat(cat_encodes, dim=1))
            f_inp = torch.stack(concat_f_inp, dim=1)
            self_attn_masks.append(obs_attn_mask)
            self_attn_inputs.append(self.obs_action_encoder(None, f_inp))

        concat_encoded_obs = []
        if obs_only:
            obs_only_attn_mask = self._get_masks_from_nans(obs_only)
            obs_only = self._copy_and_remove_nans_from_obs(obs_only, obs_only_attn_mask)
            for inputs in obs_only:
                encoded = self.observation_encoder(inputs)
                concat_encoded_obs.append(encoded)
            g_inp = torch.stack(concat_encoded_obs, dim=1)
            self_attn_masks.append(obs_only_attn_mask)
            self_attn_inputs.append(self.obs_encoder(None, g_inp))

        encoded_entity = torch.cat(self_attn_inputs, dim=1)
        encoded_state = self.self_attn(encoded_entity, self_attn_masks)

        flipped_masks = 1 - torch.cat(self_attn_masks, dim=1)
        num_agents = torch.sum(flipped_masks, dim=1, keepdim=True)
        if torch.max(num_agents).item() > self._current_max_agents:
            self._current_max_agents = torch.nn.Parameter(
                torch.as_tensor(torch.max(num_agents).item()), requires_grad=False
            )

        # num_agents will be -1 for a single agent and +1 when the current maximum is reached
        num_agents = num_agents * 2.0 / self._current_max_agents - 1

        encoding = self.linear_encoder(encoded_state)
        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        encoding = torch.cat([encoding, num_agents], dim=1)
        return encoding, memories


class Critic(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value outputs for the given obs.
        :param inputs: List of inputs as tensors.
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values.
        """
        pass


class ValueNetwork(nn.Module, Critic):
    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):

        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        self.network_body = NetworkBody(
            observation_specs, network_settings, encoded_act_size=encoded_act_size
        )
        if network_settings.memory is not None:
            encoding_size = network_settings.memory.memory_size // 2
        else:
            encoding_size = network_settings.hidden_units
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        value_outputs, critic_mem_out = self.forward(
            inputs, memories=memories, sequence_length=sequence_length
        )
        return value_outputs, critic_mem_out

    def forward(
        self,
        inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs, actions, memories, sequence_length
        )
        output = self.value_heads(encoding)
        return output, memories


class Actor(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        pass

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        """
        Returns log_probs for actions and entropies.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param actions: AgentAction of actions.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """

        pass

    @abc.abstractmethod
    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Forward pass of the Actor for inference. This is required for export to ONNX, and
        the inputs and outputs of this method should not be changed without a respective change
        in the ONNX export code.
        """
        pass


class SimpleActor(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
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
        self.network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = network_settings.hidden_units
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )

        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:

        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        return action, run_out, memories

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding, actor_mem_outs = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )

        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        run_out = {}
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return run_out

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.

        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """
        encoding, memories_out = self.network_body(
            inputs, memories=memories, sequence_length=1
        )

        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
            deterministic_cont_action_out,
            deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)
        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [
                cont_action_out,
                self.continuous_act_size_vector,
                deterministic_cont_action_out,
            ]
        if self.action_spec.discrete_size > 0:
            export_out += [
                disc_action_out,
                self.discrete_act_size_vector,
                deterministic_disc_action_out,
            ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        return tuple(export_out)


class SharedActorCritic(SimpleActor, Critic):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            observation_specs,
            network_settings,
            action_spec,
            conditional_sigma,
            tanh_squash,
        )
        self.stream_names = stream_names
        self.value_heads = ValueHeads(stream_names, self.encoding_size)

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories_out = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        return self.value_heads(encoding), memories_out


class GlobalSteps(nn.Module):
    def __init__(self):
        super().__init__()
        self.__global_step = nn.Parameter(
            torch.Tensor([0]).to(torch.int64), requires_grad=False
        )

    @property
    def current_step(self):
        return int(self.__global_step.item())

    @current_step.setter
    def current_step(self, value):
        self.__global_step[:] = value

    def increment(self, value):
        self.__global_step += value


class LearningRate(nn.Module):
    def __init__(self, lr):
        # Todo: add learning rate decay
        super().__init__()
        self.learning_rate = torch.Tensor([lr])

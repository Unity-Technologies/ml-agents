from typing import Callable, List, Dict, Tuple, Optional, Union
import abc

from mlagents.torch_utils import torch, nn

from mlagents_envs.base_env import ActionSpec, SensorSpec
from mlagents.trainers.torch.action_model import ActionModel
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.decoders import ValueHeads
from mlagents.trainers.torch.layers import LSTM, LinearEncoder
from mlagents.trainers.torch.encoders import VectorInput
from mlagents.trainers.buffer import AgentBuffer
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch.attention import ResidualSelfAttention, EntityEmbedding


ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool], torch.Tensor
]

EPSILON = 1e-7


class NetworkBody(nn.Module):
    def __init__(
        self,
        sensor_specs: List[SensorSpec],
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

        self.processors, self.embedding_sizes = ModelUtils.create_input_processors(
            sensor_specs,
            self.h_size,
            network_settings.vis_encode_type,
            normalize=self.normalize,
        )

        total_enc_size = sum(self.embedding_sizes) + encoded_act_size
        self.linear_encoder = LinearEncoder(
            total_enc_size, network_settings.num_layers, self.h_size
        )

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore

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
        encodes = []
        for idx, processor in enumerate(self.processors):
            obs_input = inputs[idx]
            processed_obs = processor(obs_input)
            encodes.append(processed_obs)

        if len(encodes) == 0:
            raise Exception("No valid inputs to network.")

        # Constants don't work in Barracuda
        if actions is not None:
            inputs = torch.cat(encodes + [actions], dim=-1)
        else:
            inputs = torch.cat(encodes, dim=-1)
        encoding = self.linear_encoder(inputs)

        if self.use_lstm:
            # Resize to (batch, sequence length, encoding size)
            encoding = encoding.reshape([-1, sequence_length, self.h_size])
            encoding, memories = self.lstm(encoding, memories)
            encoding = encoding.reshape([-1, self.m_size // 2])
        return encoding, memories


# NOTE: this class will be replaced with a multi-head attention when the time comes
class MultiInputNetworkBody(nn.Module):
    def __init__(
        self,
        sensor_specs: List[SensorSpec],
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

        encoder_input_size = self.h_size

        self.linear_encoder = LinearEncoder(
            encoder_input_size, network_settings.num_layers, self.h_size
        )

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore

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

    def q_net(
        self,
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self_attn_masks = []
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

        f_inp = torch.stack(concat_f_inp, dim=1)
        self_attn_masks.append(self._get_masks_from_nans(obs))
        encoding, memories = self.forward(
            f_inp,
            None,
            self_attn_masks,
            memories=memories,
            sequence_length=sequence_length,
        )
        return encoding, memories

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

    def value(
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


class ValueNetwork(nn.Module):
    def __init__(
        self,
        stream_names: List[str],
        sensor_specs: List[SensorSpec],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):

        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        self.network_body = NetworkBody(
            sensor_specs, network_settings, encoded_act_size=encoded_act_size
        )
        if network_settings.memory is not None:
            encoding_size = network_settings.memory.memory_size // 2
        else:
            encoding_size = network_settings.hidden_units
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

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


class CentralizedValueNetwork(ValueNetwork):
    def __init__(
        self,
        stream_names: List[str],
        observation_shapes: List[SensorSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        outputs_per_stream: int = 1,
    ):
        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        self.network_body = MultiInputNetworkBody(
            observation_shapes, network_settings, action_spec=action_spec
        )
        if network_settings.memory is not None:
            encoding_size = network_settings.memory.memory_size // 2
        else:
            encoding_size = network_settings.hidden_units
        self.value_heads = ValueHeads(stream_names, encoding_size, outputs_per_stream)

    def q_net(
        self,
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding, memories = self.network_body.q_net(
            obs, actions, memories, sequence_length
        )
        output = self.value_heads(encoding)
        return output, memories

    def value(
        self,
        obs: List[List[torch.Tensor]],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding, memories = self.network_body.value(obs, memories, sequence_length)
        output = self.value_heads(encoding)
        return output, memories

    def baseline(
        self,
        self_obs: List[List[torch.Tensor]],
        obs: List[List[torch.Tensor]],
        actions: List[AgentAction],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoding, memories = self.network_body.baseline(
            self_obs, obs, actions, memories, sequence_length
        )
        output = self.value_heads(encoding)
        return output, memories

    def forward(
        self,
        value_inputs: List[List[torch.Tensor]],
        q_inputs: List[List[torch.Tensor]],
        q_actions: List[AgentAction] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            value_inputs, q_inputs, q_actions, memories, sequence_length
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

    def get_action_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param vec_inputs: A List of vector inputs as tensors.
        :param vis_inputs: A List of visual inputs as tensors.
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
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Forward pass of the Actor for inference. This is required for export to ONNX, and
        the inputs and outputs of this method should not be changed without a respective change
        in the ONNX export code.
        """
        pass


class ActorCritic(Actor):
    @abc.abstractmethod
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

    @abc.abstractmethod
    def get_action_stats_and_value(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        critic_obs: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[
        AgentAction, ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        """
        Returns sampled actions and value estimates.
        If memory is enabled, return the memories as well.
        :param inputs: A List of vector inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, Dict of reward signal
            name to value estimate, and memories. Memories will be None if not using memory.
        """
        pass

    @abc.abstractproperty
    def memory_size(self):
        """
        Returns the size of the memory (same size used as input and output in the other
        methods) used by this Actor.
        """
        pass


class SimpleActor(nn.Module, Actor):
    def __init__(
        self,
        sensor_specs: List[SensorSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.action_spec = action_spec
        self.version_number = torch.nn.Parameter(
            torch.Tensor([2.0]), requires_grad=False
        )
        self.is_continuous_int_deprecated = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        # TODO: export list of branch sizes instead of sum
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([sum(self.action_spec.discrete_branches)]), requires_grad=False
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
        self.network_body = NetworkBody(sensor_specs, network_settings)
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
        )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def get_action_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:

        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        return action, log_probs, entropies, memories

    def forward(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.

        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """
        # This code will convert the vec and vis obs into a list of inputs for the network
        concatenated_vec_obs = vec_inputs[0]
        inputs = []
        start = 0
        end = 0
        vis_index = 0
        for i, enc in enumerate(self.network_body.processors):
            if isinstance(enc, VectorInput):
                # This is a vec_obs
                vec_size = self.network_body.embedding_sizes[i]
                end = start + vec_size
                inputs.append(concatenated_vec_obs[:, start:end])
                start = end
            else:
                inputs.append(vis_inputs[vis_index])
                vis_index += 1
        # End of code to convert the vec and vis obs into a list of inputs for the network
        encoding, memories_out = self.network_body(
            inputs, memories=memories, sequence_length=1
        )

        (
            cont_action_out,
            disc_action_out,
            action_out_deprecated,
        ) = self.action_model.get_action_out(encoding, masks)
        export_out = [self.version_number, self.memory_size_vector]
        if self.action_spec.continuous_size > 0:
            export_out += [cont_action_out, self.continuous_act_size_vector]
        if self.action_spec.discrete_size > 0:
            export_out += [disc_action_out, self.discrete_act_size_vector]
        # Only export deprecated nodes with non-hybrid action spec
        if self.action_spec.continuous_size == 0 or self.action_spec.discrete_size == 0:
            export_out += [
                action_out_deprecated,
                self.is_continuous_int_deprecated,
                self.act_size_vector_deprecated,
            ]
        return tuple(export_out)


class SharedActorCritic(SimpleActor, ActorCritic):
    def __init__(
        self,
        sensor_specs: List[SensorSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            sensor_specs, network_settings, action_spec, conditional_sigma, tanh_squash
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

    def get_stats_and_value(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        team_obs: Optional[List[List[torch.Tensor]]] = None,
        team_act: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor]]:
        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        value_outputs = self.value_heads(encoding)
        return log_probs, entropies, value_outputs

    def get_action_stats_and_value(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[
        AgentAction, ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:

        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        value_outputs = self.value_heads(encoding)
        return action, log_probs, entropies, value_outputs, memories


class SeparateActorCritic(SimpleActor, ActorCritic):
    def __init__(
        self,
        sensor_specs: List[SensorSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            sensor_specs, network_settings, action_spec, conditional_sigma, tanh_squash
        )
        self.stream_names = stream_names
        self.critic = CentralizedValueNetwork(
            stream_names, sensor_specs, network_settings, action_spec=action_spec
        )
        # self.target = CentralizedValueNetwork(
        #     stream_names, sensor_specs, network_settings, action_spec=action_spec
        # )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size + self.critic.memory_size

    def _get_actor_critic_mem(
        self, memories: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.use_lstm and memories is not None:
            # Use only the back half of memories for critic and actor
            actor_mem, critic_mem = torch.split(memories, self.memory_size // 2, dim=-1)
        else:
            critic_mem = None
            actor_mem = None
        return actor_mem, critic_mem

    def target_critic_value(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        team_obs: List[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)

        all_obs = [inputs]
        if team_obs is not None and team_obs:
            all_obs.extend(team_obs)

        value_outputs, critic_mem_out = self.critic.value(
            all_obs, memories=critic_mem, sequence_length=sequence_length
        )

        # if mar_value_outputs is None:
        #    mar_value_outputs = value_outputs

        if actor_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem, critic_mem_out], dim=-1)
        else:
            memories_out = None
        return value_outputs, memories_out

    def critic_value(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        team_obs: List[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)

        all_obs = [inputs]
        if team_obs is not None and team_obs:
            all_obs.extend(team_obs)

        value_outputs, critic_mem_out = self.critic.value(
            all_obs, memories=critic_mem, sequence_length=sequence_length
        )

        # if mar_value_outputs is None:
        #    mar_value_outputs = value_outputs

        if actor_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem, critic_mem_out], dim=-1)
        else:
            memories_out = None
        return value_outputs, memories_out

    def target_critic_pass(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        team_obs: List[List[torch.Tensor]] = None,
        team_act: List[AgentAction] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)

        all_obs = [inputs]
        if team_obs is not None and team_obs:
            all_obs.extend(team_obs)
        all_acts = [actions]
        if team_act is not None and team_act:
            all_acts.extend(team_act)

        baseline_outputs, _ = self.critic.baseline(
            inputs,
            team_obs,
            team_act,
            memories=critic_mem,
            sequence_length=sequence_length,
        )

        value_outputs, critic_mem_out = self.critic.q_net(
            all_obs, all_acts, memories=critic_mem, sequence_length=sequence_length
        )

        # if mar_value_outputs is None:
        #    mar_value_outputs = value_outputs

        if actor_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem, critic_mem_out], dim=-1)
        else:
            memories_out = None
        return value_outputs, baseline_outputs, memories_out

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        team_obs: List[List[torch.Tensor]] = None,
        team_act: List[AgentAction] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)

        all_obs = [inputs]
        if team_obs is not None and team_obs:
            all_obs.extend(team_obs)
        all_acts = [actions]
        if team_act is not None and team_act:
            all_acts.extend(team_act)

        baseline_outputs, critic_mem_out = self.critic.baseline(
            inputs,
            team_obs,
            team_act,
            memories=critic_mem,
            sequence_length=sequence_length,
        )

        # q_out, critic_mem_out = self.critic.q_net(
        #     all_obs, all_acts, memories=critic_mem, sequence_length=sequence_length
        # )

        # if mar_value_outputs is None:
        #    mar_value_outputs = value_outputs

        if actor_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem, critic_mem_out], dim=-1)
        else:
            memories_out = None
        return baseline_outputs, memories_out

    def get_stats_and_value(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        team_obs: Optional[List[List[torch.Tensor]]] = None,
        team_act: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor]]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        encoding, actor_mem_outs = self.network_body(
            inputs, memories=actor_mem, sequence_length=sequence_length
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)

        baseline_outputs, _ = self.critic_pass(
            inputs,
            actions,
            memories=critic_mem,
            sequence_length=sequence_length,
            team_obs=team_obs,
            team_act=team_act,
        )
        value_outputs, _ = self.target_critic_value(
            inputs,
            memories=critic_mem,
            sequence_length=sequence_length,
            team_obs=team_obs,
        )

        return log_probs, entropies, baseline_outputs, value_outputs

    def get_action_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        action, log_probs, entropies, actor_mem_out = super().get_action_stats(
            inputs, masks=masks, memories=actor_mem, sequence_length=sequence_length
        )
        if critic_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem_out, critic_mem], dim=-1)
        else:
            memories_out = None
        return action, log_probs, entropies, memories_out

    def get_action_stats_and_value(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        critic_obs: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[
        AgentAction, ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        encoding, actor_mem_outs = self.network_body(
            inputs, memories=actor_mem, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        all_net_inputs = [inputs]
        if critic_obs is not None:
            all_net_inputs.extend(critic_obs)
        value_outputs, critic_mem_outs = self.critic(
            all_net_inputs, memories=critic_mem, sequence_length=sequence_length
        )
        if self.use_lstm:
            mem_out = torch.cat([actor_mem_outs, critic_mem_outs], dim=-1)
        else:
            mem_out = None
        return action, log_probs, entropies, value_outputs, mem_out

    def update_normalization(self, buffer: AgentBuffer) -> None:
        super().update_normalization(buffer)
        self.critic.network_body.update_normalization(buffer)


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

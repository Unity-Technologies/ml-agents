from typing import Callable, List, Dict, Tuple, Optional, Union
import abc

from mlagents.torch_utils import torch, nn

from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch.action_model import ActionModel
from mlagents.trainers.torch.agent_action import AgentAction
from mlagents.trainers.torch.action_log_probs import ActionLogProbs
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.decoders import ValueHeads
from mlagents.trainers.torch.layers import LSTM, LinearEncoder
from mlagents.trainers.torch.model_serialization import exporting_to_onnx

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
EncoderFunction = Callable[
    [torch.Tensor, int, ActivationFunction, int, str, bool], torch.Tensor
]

EPSILON = 1e-7


class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
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

        (
            self.visual_processors,
            self.vector_processors,
            encoder_input_size,
        ) = ModelUtils.create_input_processors(
            observation_shapes,
            self.h_size,
            network_settings.vis_encode_type,
            normalize=self.normalize,
        )
        total_enc_size = encoder_input_size + encoded_act_size
        self.linear_encoder = LinearEncoder(
            total_enc_size, network_settings.num_layers, self.h_size
        )

        if self.use_lstm:
            self.lstm = LSTM(self.h_size, self.m_size)
        else:
            self.lstm = None  # type: ignore

    def update_normalization(self, vec_inputs: List[torch.Tensor]) -> None:
        for vec_input, vec_enc in zip(vec_inputs, self.vector_processors):
            vec_enc.update_normalization(vec_input)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        if self.normalize:
            for n1, n2 in zip(self.vector_processors, other_network.vector_processors):
                n1.copy_normalization(n2)

    @property
    def memory_size(self) -> int:
        return self.lstm.memory_size if self.use_lstm else 0

    def forward(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encodes = []
        for idx, processor in enumerate(self.vector_processors):
            vec_input = vec_inputs[idx]
            processed_vec = processor(vec_input)
            encodes.append(processed_vec)

        for idx, processor in enumerate(self.visual_processors):
            vis_input = vis_inputs[idx]
            if not exporting_to_onnx.is_exporting():
                vis_input = vis_input.permute([0, 3, 1, 2])
            processed_vis = processor(vis_input)
            encodes.append(processed_vis)

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


class ValueNetwork(nn.Module):
    def __init__(
        self,
        stream_names: List[str],
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        encoded_act_size: int = 0,
        outputs_per_stream: int = 1,
    ):

        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        self.network_body = NetworkBody(
            observation_shapes, network_settings, encoded_act_size=encoded_act_size
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
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            vec_inputs, vis_inputs, actions, memories, sequence_length
        )
        output = self.value_heads(encoding)
        return output, memories


class Actor(abc.ABC):
    @abc.abstractmethod
    def update_normalization(self, vector_obs: List[torch.Tensor]) -> None:
        """
        Updates normalization of Actor based on the provided List of vector obs.
        :param vector_obs: A List of vector obs as tensors.
        """
        pass

    def get_action_stats(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
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
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value outputs for the given obs.
        :param vec_inputs: List of vector inputs as tensors.
        :param vis_inputs: List of visual inputs as tensors.
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values.
        """
        pass

    @abc.abstractmethod
    def get_action_stats_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[
        AgentAction, ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        """
        Returns sampled actions and value estimates.
        If memory is enabled, return the memories as well.
        :param vec_inputs: A List of vector inputs as tensors.
        :param vis_inputs: A List of visual inputs as tensors.
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
        observation_shapes: List[Tuple[int, ...]],
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
        self.network_body = NetworkBody(observation_shapes, network_settings)
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

    def update_normalization(self, vector_obs: List[torch.Tensor]) -> None:
        self.network_body.update_normalization(vector_obs)

    def get_action_stats(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:

        encoding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
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
        encoding, memories_out = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=1
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
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            observation_shapes,
            network_settings,
            action_spec,
            conditional_sigma,
            tanh_squash,
        )
        self.stream_names = stream_names
        self.value_heads = ValueHeads(stream_names, self.encoding_size)

    def critic_pass(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories_out = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        return self.value_heads(encoding), memories_out

    def get_stats_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor]]:
        encoding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        value_outputs = self.value_heads(encoding)
        return log_probs, entropies, value_outputs

    def get_action_stats_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[
        AgentAction, ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:

        encoding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        value_outputs = self.value_heads(encoding)
        return action, log_probs, entropies, value_outputs, memories


class SeparateActorCritic(SimpleActor, ActorCritic):
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        self.use_lstm = network_settings.memory is not None
        super().__init__(
            observation_shapes,
            network_settings,
            action_spec,
            conditional_sigma,
            tanh_squash,
        )
        self.stream_names = stream_names
        self.critic = ValueNetwork(stream_names, observation_shapes, network_settings)

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

    def critic_pass(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        value_outputs, critic_mem_out = self.critic(
            vec_inputs, vis_inputs, memories=critic_mem, sequence_length=sequence_length
        )
        if actor_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem, critic_mem_out], dim=-1)
        else:
            memories_out = None
        return value_outputs, memories_out

    def get_stats_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor]]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        encoding, actor_mem_outs = self.network_body(
            vec_inputs, vis_inputs, memories=actor_mem, sequence_length=sequence_length
        )
        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        value_outputs, critic_mem_outs = self.critic(
            vec_inputs, vis_inputs, memories=critic_mem, sequence_length=sequence_length
        )

        return log_probs, entropies, value_outputs

    def get_action_stats(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[AgentAction, ActionLogProbs, torch.Tensor, torch.Tensor]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        action, log_probs, entropies, actor_mem_out = super().get_action_stats(
            vec_inputs,
            vis_inputs,
            masks=masks,
            memories=actor_mem,
            sequence_length=sequence_length,
        )
        if critic_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem_out, critic_mem], dim=-1)
        else:
            memories_out = None
        return action, log_probs, entropies, memories_out

    def get_action_stats_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[
        AgentAction, ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor
    ]:
        actor_mem, critic_mem = self._get_actor_critic_mem(memories)
        encoding, actor_mem_outs = self.network_body(
            vec_inputs, vis_inputs, memories=actor_mem, sequence_length=sequence_length
        )
        action, log_probs, entropies = self.action_model(encoding, masks)
        value_outputs, critic_mem_outs = self.critic(
            vec_inputs, vis_inputs, memories=critic_mem, sequence_length=sequence_length
        )
        if self.use_lstm:
            mem_out = torch.cat([actor_mem_outs, critic_mem_outs], dim=-1)
        else:
            mem_out = None
        return action, log_probs, entropies, value_outputs, mem_out

    def update_normalization(self, vector_obs: List[torch.Tensor]) -> None:
        super().update_normalization(vector_obs)
        self.critic.network_body.update_normalization(vector_obs)


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

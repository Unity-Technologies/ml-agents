from typing import Callable, List, Dict, Tuple, Optional
import abc

from mlagents.torch_utils import torch, nn

from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
    DistInstance,
)
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

        self.visual_processors, self.vector_processors, encoder_input_size = ModelUtils.create_input_processors(
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

    @abc.abstractmethod
    def sample_action(self, dists: List[DistInstance]) -> List[torch.Tensor]:
        """
        Takes a List of Distribution iinstances and samples an action from each.
        """
        pass

    @abc.abstractmethod
    def get_dists(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[DistInstance], Optional[torch.Tensor]]:
        """
        Returns distributions from this Actor, from which actions can be sampled.
        If memory is enabled, return the memories as well.
        :param vec_inputs: A List of vector inputs as tensors.
        :param vis_inputs: A List of visual inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of a List of action distribution instances, and memories.
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
    ) -> Tuple[torch.Tensor, int, int, int, int]:
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
    def get_dist_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[DistInstance], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns distributions, from which actions can be sampled, and value estimates.
        If memory is enabled, return the memories as well.
        :param vec_inputs: A List of vector inputs as tensors.
        :param vis_inputs: A List of visual inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of a List of action distribution instances, a Dict of reward signal
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
        self.version_number = torch.nn.Parameter(torch.Tensor([2.0]))
        self.is_continuous_int = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.is_continuous())])
        )
        self.act_size_vector = torch.nn.Parameter(
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

        if self.action_spec.is_continuous():
            self.distribution = GaussianDistribution(
                self.encoding_size,
                self.action_spec.continuous_size,
                conditional_sigma=conditional_sigma,
                tanh_squash=tanh_squash,
            )
        else:
            self.distribution = MultiCategoricalDistribution(
                self.encoding_size, self.action_spec.discrete_branches
            )
        # During training, clipping is done in TorchPolicy, but we need to clip before ONNX
        # export as well.
        self._clip_action_on_export = not tanh_squash

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, vector_obs: List[torch.Tensor]) -> None:
        self.network_body.update_normalization(vector_obs)

    def sample_action(self, dists: List[DistInstance]) -> List[torch.Tensor]:
        actions = []
        for action_dist in dists:
            action = action_dist.sample()
            actions.append(action)
        return actions

    def get_dists(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[DistInstance], Optional[torch.Tensor]]:
        encoding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        if self.action_spec.is_continuous():
            dists = self.distribution(encoding)
        else:
            dists = self.distribution(encoding, masks)

        return dists, memories

    def forward(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.
        """
        dists, _ = self.get_dists(vec_inputs, vis_inputs, masks, memories, 1)
        if self.action_spec.is_continuous():
            action_list = self.sample_action(dists)
            action_out = torch.stack(action_list, dim=-1)
            if self._clip_action_on_export:
                action_out = torch.clamp(action_out, -3, 3) / 3
        else:
            action_out = torch.cat([dist.all_log_prob() for dist in dists], dim=1)
        return (
            action_out,
            self.version_number,
            torch.Tensor([self.network_body.memory_size]),
            self.is_continuous_int,
            self.act_size_vector,
        )


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

    def get_dist_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[DistInstance], Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        if self.action_spec.is_continuous():
            dists = self.distribution(encoding)
        else:
            dists = self.distribution(encoding, masks=masks)

        value_outputs = self.value_heads(encoding)
        return dists, value_outputs, memories


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
        # Give the Actor only half the memories. Note we previously validate
        # that memory_size must be a multiple of 4.
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

    def critic_pass(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        actor_mem, critic_mem = None, None
        if self.use_lstm:
            # Use only the back half of memories for critic
            actor_mem, critic_mem = torch.split(memories, self.memory_size // 2, -1)
        value_outputs, critic_mem_out = self.critic(
            vec_inputs, vis_inputs, memories=critic_mem, sequence_length=sequence_length
        )
        if actor_mem is not None:
            # Make memories with the actor mem unchanged
            memories_out = torch.cat([actor_mem, critic_mem_out], dim=-1)
        else:
            memories_out = None
        return value_outputs, memories_out

    def get_dist_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[DistInstance], Dict[str, torch.Tensor], torch.Tensor]:
        if self.use_lstm:
            # Use only the back half of memories for critic and actor
            actor_mem, critic_mem = torch.split(memories, self.memory_size // 2, dim=-1)
        else:
            critic_mem = None
            actor_mem = None
        dists, actor_mem_outs = self.get_dists(
            vec_inputs,
            vis_inputs,
            memories=actor_mem,
            sequence_length=sequence_length,
            masks=masks,
        )
        value_outputs, critic_mem_outs = self.critic(
            vec_inputs, vis_inputs, memories=critic_mem, sequence_length=sequence_length
        )
        if self.use_lstm:
            mem_out = torch.cat([actor_mem_outs, critic_mem_outs], dim=-1)
        else:
            mem_out = None
        return dists, value_outputs, mem_out

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

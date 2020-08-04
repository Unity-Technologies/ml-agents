from typing import Callable, List, Dict, Tuple, Optional
import attr
import abc

import torch
from torch import nn

from mlagents_envs.base_env import ActionType
from mlagents.trainers.torch.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
    DistInstance,
)
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch.utils import ModelUtils
from mlagents.trainers.torch.decoders import ValueHeads

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

        self.visual_encoders, self.vector_encoders = ModelUtils.create_encoders(
            observation_shapes,
            self.h_size,
            network_settings.num_layers,
            network_settings.vis_encode_type,
            unnormalized_inputs=encoded_act_size,
            normalize=self.normalize,
        )

        if self.use_lstm:
            self.lstm = nn.LSTM(self.h_size, self.m_size // 2, 1)
        else:
            self.lstm = None

    def update_normalization(self, vec_inputs: List[torch.Tensor]) -> None:
        for vec_input, vec_enc in zip(vec_inputs, self.vector_encoders):
            vec_enc.update_normalization(vec_input)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        if self.normalize:
            for n1, n2 in zip(self.vector_encoders, other_network.vector_encoders):
                n1.copy_normalization(n2)

    def forward(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vec_encodes = []
        for idx, encoder in enumerate(self.vector_encoders):
            vec_input = vec_inputs[idx]
            if actions is not None:
                hidden = encoder(vec_input, actions)
            else:
                hidden = encoder(vec_input)
            vec_encodes.append(hidden)

        vis_encodes = []
        for idx, encoder in enumerate(self.visual_encoders):
            vis_input = vis_inputs[idx]
            vis_input = vis_input.permute([0, 3, 1, 2])
            hidden = encoder(vis_input)
            vis_encodes.append(hidden)

        if len(vec_encodes) > 0 and len(vis_encodes) > 0:
            vec_encodes_tensor = torch.stack(vec_encodes, dim=-1).sum(dim=-1)
            vis_encodes_tensor = torch.stack(vis_encodes, dim=-1).sum(dim=-1)
            encoding = torch.stack(
                [vec_encodes_tensor, vis_encodes_tensor], dim=-1
            ).sum(dim=-1)
        elif len(vec_encodes) > 0:
            encoding = torch.stack(vec_encodes, dim=-1).sum(dim=-1)
        elif len(vis_encodes) > 0:
            encoding = torch.stack(vis_encodes, dim=-1).sum(dim=-1)
        else:
            raise Exception("No valid inputs to network.")

        if self.use_lstm:
            encoding = encoding.view([sequence_length, -1, self.h_size])
            memories = torch.split(memories, self.m_size // 2, dim=-1)
            encoding, memories = self.lstm(
                encoding.contiguous(),
                (memories[0].contiguous(), memories[1].contiguous()),
            )
            encoding = encoding.view([-1, self.m_size // 2])
            memories = torch.cat(memories, dim=-1)
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
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
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
    ) -> Dict[str, torch.Tensor]:
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


class SimpleActor(nn.Module, Actor):
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        act_type: ActionType,
        act_size: List[int],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.act_type = act_type
        self.act_size = act_size
        self.version_number = torch.nn.Parameter(torch.Tensor([2.0]))
        self.memory_size = torch.nn.Parameter(torch.Tensor([0]))
        self.is_continuous_int = torch.nn.Parameter(
            torch.Tensor([int(act_type == ActionType.CONTINUOUS)])
        )
        self.act_size_vector = torch.nn.Parameter(torch.Tensor(act_size))
        self.network_body = NetworkBody(observation_shapes, network_settings)
        if network_settings.memory is not None:
            self.encoding_size = network_settings.memory.memory_size // 2
        else:
            self.encoding_size = network_settings.hidden_units
        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(
                self.encoding_size,
                act_size[0],
                conditional_sigma=conditional_sigma,
                tanh_squash=tanh_squash,
            )
        else:
            self.distribution = MultiCategoricalDistribution(
                self.encoding_size, act_size
            )

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
        if self.act_type == ActionType.CONTINUOUS:
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
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.
        """
        dists, _ = self.get_dists(
            vec_inputs, vis_inputs, masks, memories, sequence_length
        )
        action_list = self.sample_action(dists)
        sampled_actions = torch.stack(action_list, dim=-1)
        return (
            sampled_actions,
            dists[0].pdf(sampled_actions),
            self.version_number,
            self.memory_size,
            self.is_continuous_int,
            self.act_size_vector,
        )


class SharedActorCritic(SimpleActor, ActorCritic):
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        act_type: ActionType,
        act_size: List[int],
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__(
            observation_shapes,
            network_settings,
            act_type,
            act_size,
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
    ) -> Dict[str, torch.Tensor]:
        encoding, _ = self.network_body(vec_inputs, vis_inputs, memories=memories)
        return self.value_heads(encoding)

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
        if self.act_type == ActionType.CONTINUOUS:
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
        act_type: ActionType,
        act_size: List[int],
        stream_names: List[str],
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        # Give the Actor only half the memories. Note we previously validate
        # that memory_size must be a multiple of 4.
        self.use_lstm = network_settings.memory is not None
        if network_settings.memory is not None:
            self.half_mem_size = network_settings.memory.memory_size // 2
            new_memory_settings = attr.evolve(
                network_settings.memory, memory_size=self.half_mem_size
            )
            use_network_settings = attr.evolve(
                network_settings, memory=new_memory_settings
            )
        else:
            use_network_settings = network_settings
            self.half_mem_size = 0
        super().__init__(
            observation_shapes,
            use_network_settings,
            act_type,
            act_size,
            conditional_sigma,
            tanh_squash,
        )
        self.stream_names = stream_names
        self.critic = ValueNetwork(
            stream_names, observation_shapes, use_network_settings
        )

    def critic_pass(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if memories is not None:
            # Use only the back half of memories for critic
            _, critic_mem = torch.split(memories, self.half_mem_size, -1)
        else:
            critic_mem = None
        value_outputs, _memories = self.critic(
            vec_inputs, vis_inputs, memories=critic_mem
        )
        return value_outputs

    def get_dist_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[DistInstance], Dict[str, torch.Tensor], torch.Tensor]:
        if memories is not None:
            # Use only the back half of memories for critic and actor
            actor_mem, critic_mem = torch.split(memories, self.half_mem_size, dim=-1)
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
            mem_out = torch.cat([actor_mem_outs, critic_mem_outs], dim=1)
        else:
            mem_out = None
        return dists, value_outputs, mem_out


class GlobalSteps(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_step = torch.Tensor([0])

    def increment(self, value):
        self.global_step += value


class LearningRate(nn.Module):
    def __init__(self, lr):
        # Todo: add learning rate decay
        super().__init__()
        self.learning_rate = torch.Tensor([lr])

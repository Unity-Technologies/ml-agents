from typing import Callable, List, Dict, Tuple, Optional

import torch
from torch import nn

from mlagents_envs.base_env import ActionType
from mlagents.trainers.torch.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
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

    def update_normalization(self, vec_inputs):
        for vec_input, vec_enc in zip(vec_inputs, self.vector_encoders):
            vec_enc.update_normalization(vec_input)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        if self.normalize:
            for n1, n2 in zip(self.vector_encoders, other_network.vector_encoders):
                n1.copy_normalization(n2)

    def forward(
        self,
        vec_inputs: torch.Tensor,
        vis_inputs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vec_embeds = []
        for idx, encoder in enumerate(self.vector_encoders):
            vec_input = vec_inputs[idx]
            if actions is not None:
                hidden = encoder(vec_input, actions)
            else:
                hidden = encoder(vec_input)
            vec_embeds.append(hidden)

        vis_embeds = []
        for idx, encoder in enumerate(self.visual_encoders):
            vis_input = vis_inputs[idx]
            vis_input = vis_input.permute([0, 3, 1, 2])
            hidden = encoder(vis_input)
            vis_embeds.append(hidden)

        # embedding = vec_embeds[0]
        if len(vec_embeds) > 0 and len(vis_embeds) > 0:
            vec_embeds_tensor = torch.stack(vec_embeds, dim=-1).sum(dim=-1)
            vis_embeds_tensor = torch.stack(vis_embeds, dim=-1).sum(dim=-1)
            embedding = torch.stack([vec_embeds_tensor, vis_embeds_tensor], dim=-1).sum(
                dim=-1
            )
        elif len(vec_embeds) > 0:
            embedding = torch.stack(vec_embeds, dim=-1).sum(dim=-1)
        elif len(vis_embeds) > 0:
            embedding = torch.stack(vis_embeds, dim=-1).sum(dim=-1)
        else:
            raise Exception("No valid inputs to network.")

        if self.use_lstm:
            embedding = embedding.view([sequence_length, -1, self.h_size])
            memories = torch.split(memories, self.m_size // 2, dim=-1)
            embedding, memories = self.lstm(
                embedding.contiguous(),
                (memories[0].contiguous(), memories[1].contiguous()),
            )
            embedding = embedding.view([-1, self.m_size // 2])
            memories = torch.cat(memories, dim=-1)
        return embedding, memories


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
        self.value_heads = ValueHeads(
            stream_names, network_settings.hidden_units, outputs_per_stream
        )

    def forward(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, actions, memories, sequence_length
        )
        output = self.value_heads(embedding)
        return output, memories


class Actor(nn.Module):
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
        self.is_continuous_int = torch.nn.Parameter(torch.Tensor([1]))
        self.act_size_vector = torch.nn.Parameter(torch.Tensor(act_size))
        self.network_body = NetworkBody(observation_shapes, network_settings)
        if network_settings.memory is not None:
            self.embedding_size = network_settings.memory.memory_size // 2
        else:
            self.embedding_size = network_settings.hidden_units
        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(
                self.embedding_size,
                act_size[0],
                conditional_sigma=conditional_sigma,
                tanh_squash=tanh_squash,
            )
        else:
            self.distribution = MultiCategoricalDistribution(
                self.embedding_size, act_size
            )

    def update_normalization(self, vector_obs):
        self.network_body.update_normalization(vector_obs)

    def sample_action(self, dists):
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns distributions from this Actor, from which actions can be sampled.
        If memory is enabled, return the memories as well.
        """
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        if self.act_type == ActionType.CONTINUOUS:
            dists = self.distribution(embedding)
        else:
            dists = self.distribution(embedding, masks)

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


class ActorCritic(Actor):
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
        self.value_heads = ValueHeads(stream_names, self.embedding_size)

    def critic_pass(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedding, _ = self.network_body(vec_inputs, vis_inputs, memories=memories)
        return self.value_heads(embedding)

    def get_dist_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        if self.act_type == ActionType.CONTINUOUS:
            dists = self.distribution(embedding)
        else:
            dists = self.distribution(embedding, masks=masks)

        value_outputs = self.value_heads(embedding)
        return dists, value_outputs, memories


class SeparateActorCritic(ActorCritic):
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
            stream_names,
            conditional_sigma,
            tanh_squash,
        )
        self.critic = ValueNetwork(stream_names, observation_shapes, network_settings)

    def critic_pass(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        value_outputs, _memories = self.critic(
            vec_inputs, vis_inputs, memories=memories
        )
        return value_outputs

    def get_dist_and_value(
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        dists, memories = self.get_dists(
            vec_inputs,
            vis_inputs,
            memories=memories,
            sequence_length=sequence_length,
            masks=masks,
        )
        # TODO: Feed critic memories into critic
        value_outputs, _ = self.critic(
            vec_inputs, vis_inputs, memories=memories, sequence_length=sequence_length
        )
        return dists, value_outputs, memories


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

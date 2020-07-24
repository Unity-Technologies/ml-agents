from typing import Callable, NamedTuple, List, Dict, Tuple

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


class NormalizerTensors(NamedTuple):
    steps: torch.Tensor
    running_mean: torch.Tensor
    running_variance: torch.Tensor


class NetworkBody(nn.Module):
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
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
            self.visual_encoders,
            self.vector_encoders,
            self.vector_normalizers,
        ) = ModelUtils.create_encoders(
            observation_shapes,
            self.h_size,
            network_settings.num_layers,
            network_settings.vis_encode_type,
            action_size=0,
        )

        if self.use_lstm:
            self.lstm = nn.LSTM(self.h_size, self.m_size // 2, 1)
        else:
            self.lstm = None

    def update_normalization(self, vec_inputs):
        if self.normalize:
            for idx, vec_input in enumerate(vec_inputs):
                self.vector_normalizers[idx].update(vec_input)

    def copy_normalization(self, other_network: "NetworkBody") -> None:
        if self.normalize:
            for n1, n2 in zip(
                self.vector_normalizers, other_network.vector_normalizers
            ):
                n1.copy_from(n2)

    def forward(self, vec_inputs, vis_inputs, memories=None, sequence_length=1):
        vec_embeds = []
        for idx, encoder in enumerate(self.vector_encoders):
            vec_input = vec_inputs[idx]
            if self.normalize:
                vec_input = self.vector_normalizers[idx](vec_input)
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


class QNetwork(NetworkBody):
    def __init__(  # pylint: disable=W0231
        self,
        stream_names: List[str],
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        act_type: ActionType,
        act_size: List[int],
    ):
        # This is not a typo, we want to call __init__ of nn.Module
        nn.Module.__init__(self)
        self.normalize = network_settings.normalize
        self.use_lstm = network_settings.memory is not None
        self.h_size = network_settings.hidden_units
        self.m_size = (
            network_settings.memory.memory_size
            if network_settings.memory is not None
            else 0
        )

        (
            self.visual_encoders,
            self.vector_encoders,
            self.vector_normalizers,
        ) = ModelUtils.create_encoders(
            observation_shapes,
            self.h_size,
            network_settings.num_layers,
            network_settings.vis_encode_type,
            action_size=sum(act_size) if act_type == ActionType.CONTINUOUS else 0,
        )

        if self.use_lstm:
            self.lstm = nn.LSTM(self.h_size, self.m_size // 2, 1)
        else:
            self.lstm = None
        if act_type == ActionType.DISCRETE:
            self.q_heads = ValueHeads(
                stream_names, network_settings.hidden_units, sum(act_size)
            )
        else:
            self.q_heads = ValueHeads(stream_names, network_settings.hidden_units)

    def forward(  # pylint: disable=W0221
        self,
        vec_inputs: List[torch.Tensor],
        vis_inputs: List[torch.Tensor],
        memories: torch.Tensor = None,
        sequence_length: int = 1,
        actions: torch.Tensor = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        vec_embeds = []
        for i, (enc, norm) in enumerate(
            zip(self.vector_encoders, self.vector_normalizers)
        ):
            vec_input = vec_inputs[i]
            if self.normalize:
                vec_input = norm(vec_input)
            if actions is not None:
                hidden = enc(torch.cat([vec_input, actions], dim=-1))
            else:
                hidden = enc(vec_input)
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

        if self.lstm is not None:
            embedding = embedding.view([sequence_length, -1, self.h_size])
            memories_tensor = torch.split(memories, self.m_size // 2, dim=-1)
            embedding, memories = self.lstm(embedding, memories_tensor)
            embedding = embedding.view([-1, self.m_size // 2])
            memories = torch.cat(memories_tensor, dim=-1)

        output, _ = self.q_heads(embedding)
        return output, memories


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        act_type: ActionType,
        act_size: List[int],
        stream_names: List[str],
        separate_critic: bool,
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
        self.separate_critic = separate_critic
        self.network_body = NetworkBody(observation_shapes, network_settings)
        if network_settings.memory is not None:
            embedding_size = network_settings.memory.memory_size // 2
        else:
            embedding_size = network_settings.hidden_units
        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(
                embedding_size,
                act_size[0],
                conditional_sigma=conditional_sigma,
                tanh_squash=tanh_squash,
            )
        else:
            self.distribution = MultiCategoricalDistribution(embedding_size, act_size)
        if separate_critic:
            self.critic = Critic(stream_names, observation_shapes, network_settings)
        else:
            self.stream_names = stream_names
            self.value_heads = ValueHeads(stream_names, embedding_size)

    def update_normalization(self, vector_obs):
        self.network_body.update_normalization(vector_obs)
        if self.separate_critic:
            self.critic.network_body.update_normalization(vector_obs)

    def critic_pass(self, vec_inputs, vis_inputs, memories=None):
        if self.separate_critic:
            return self.critic(vec_inputs, vis_inputs)
        else:
            embedding, _ = self.network_body(vec_inputs, vis_inputs, memories=memories)
            return self.value_heads(embedding)

    def sample_action(self, dists):
        actions = []
        for action_dist in dists:
            action = action_dist.sample()
            actions.append(action)
        return actions

    def get_probs_and_entropy(self, action_list, dists):
        log_probs = []
        all_probs = []
        entropies = []
        for action, action_dist in zip(action_list, dists):
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)
            entropies.append(action_dist.entropy())
            if self.act_type == ActionType.DISCRETE:
                all_probs.append(action_dist.all_log_prob())
        log_probs = torch.stack(log_probs, dim=-1)
        entropies = torch.stack(entropies, dim=-1)
        if self.act_type == ActionType.CONTINUOUS:
            log_probs = log_probs.squeeze(-1)
            entropies = entropies.squeeze(-1)
            all_probs = None
        else:
            all_probs = torch.cat(all_probs, dim=-1)
        return log_probs, entropies, all_probs

    def get_dist_and_value(
        self, vec_inputs, vis_inputs, masks=None, memories=None, sequence_length=1
    ):
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, memories, sequence_length
        )
        if self.act_type == ActionType.CONTINUOUS:
            dists = self.distribution(embedding)
        else:
            dists = self.distribution(embedding, masks=masks)
        if self.separate_critic:
            value_outputs = self.critic(vec_inputs, vis_inputs)
        else:
            value_outputs = self.value_heads(embedding)
        return dists, value_outputs, memories

    def forward(
        self, vec_inputs, vis_inputs=None, masks=None, memories=None, sequence_length=1
    ):
        embedding, memories = self.network_body(
            vec_inputs, vis_inputs, memories, sequence_length
        )
        dists, value_outputs, memories = self.get_dist_and_value(
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


class Critic(nn.Module):
    def __init__(
        self,
        stream_names: List[str],
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
    ):
        super().__init__()
        self.network_body = NetworkBody(observation_shapes, network_settings)
        self.stream_names = stream_names
        self.value_heads = ValueHeads(stream_names, network_settings.hidden_units)

    def forward(self, vec_inputs, vis_inputs):
        embedding, _ = self.network_body(vec_inputs, vis_inputs)
        return self.value_heads(embedding)


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

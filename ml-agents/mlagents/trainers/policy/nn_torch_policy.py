from typing import Any, Dict
import numpy as np
import torch
from mlagents_envs.base_env import DecisionSteps
from torch import nn
from mlagents.tf_utils import tf
from mlagents_envs.timers import timed
from mlagents.trainers.trajectory import SplitObservations
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.models import EncoderType
from mlagents.trainers.models_torch import (
    ActionType,
    VectorEncoder,
    ValueHeads,
    Normalizer,
    ModelUtils,
)
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.distributions_torch import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)

EPSILON = 1e-7  # Small value to avoid divide by zero


class NetworkBody(nn.Module):
    def __init__(
        self,
        vector_sizes,
        visual_sizes,
        h_size,
        normalize,
        num_layers,
        m_size,
        vis_encode_type,
        use_lstm,
    ):
        super(NetworkBody, self).__init__()
        self.normalize = normalize
        self.visual_encoders = []
        self.vector_encoders = []
        self.vector_normalizers = []
        self.use_lstm = use_lstm
        self.h_size = h_size
        self.m_size = m_size

        visual_encoder = ModelUtils.get_encoder_for_type(vis_encode_type)
        for vector_size in vector_sizes:
            self.vector_normalizers.append(Normalizer(vector_size))
            self.vector_encoders.append(VectorEncoder(vector_size, h_size, num_layers))
        for visual_size in visual_sizes:
            self.visual_encoders.append(visual_encoder(visual_size))

        if use_lstm:
            self.lstm = nn.LSTM(h_size, h_size, 1)

    def clear_memory(self, batch_size):
        self.memory = (
            torch.zeros(1, batch_size, self.m_size),
            torch.zeros(1, batch_size, self.m_size),
        )

    def update_normalization(self, inputs):
        if self.normalize:
            self.normalizer.update(inputs)

    def forward(self, vec_inputs, vis_inputs):
        vec_embeds = []
        for idx, encoder in enumerate(self.vector_encoders):
            vec_input = vec_inputs[idx]
            if self.normalize:
                vec_input = self.normalizers[idx](vec_inputs[idx])
            hidden = encoder(vec_input)
            vec_embeds.append(hidden)

        vis_embeds = []
        for idx, encoder in enumerate(self.visual_encoders):
            hidden = encoder(vis_inputs[idx])
            vis_embeds.append(hidden)

        vec_embeds = torch.cat(vec_embeds)
        vis_embeds = torch.cat(vis_embeds)
        embedding = torch.cat([vec_embeds, vis_embeds])
        if self.use_lstm:
            embedding, self.memory = self.lstm(embedding, self.memory)
        return embedding


class Actor(nn.Module):
    def __init__(
        self,
        h_size,
        vector_sizes,
        visual_sizes,
        act_size,
        normalize,
        num_layers,
        m_size,
        vis_encode_type,
        act_type,
        use_lstm,
    ):
        super(Actor, self).__init__()
        self.act_type = act_type
        self.act_size = act_size
        self.network_body = NetworkBody(
            vector_sizes,
            visual_sizes,
            h_size,
            normalize,
            num_layers,
            m_size,
            vis_encode_type,
            use_lstm,
        )
        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(h_size, act_size)
        else:
            self.distribution = MultiCategoricalDistribution(h_size, act_size)

    def forward(self, vec_inputs, vis_inputs, masks=None):
        embedding = self.network_body(vec_inputs, vis_inputs)
        if self.act_type == ActionType.CONTINUOUS:
            dist = self.distribution(embedding)
        else:
            dist = self.distribution(embedding, masks=masks)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        stream_names,
        h_size,
        vector_sizes,
        visual_sizes,
        normalize,
        num_layers,
        m_size,
        vis_encode_type,
        use_lstm,
    ):
        super(Critic, self).__init__()
        self.stream_names = stream_names
        self.network_body = NetworkBody(
            vector_sizes,
            visual_sizes,
            h_size,
            normalize,
            num_layers,
            m_size,
            vis_encode_type,
            use_lstm,
        )
        self.value_heads = ValueHeads(stream_names, h_size)

    def forward(self, vec_inputs, vis_inputs):
        embedding = self.network_body(vec_inputs, vis_inputs)
        return self.value_heads(embedding)


class NNPolicy(TorchPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        load: bool,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous action spaces, as well as recurrent networks.
        :param seed: Random seed.
        :param brain: Assigned BrainParameters object.
        :param trainer_params: Defined training parameters.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        :param tanh_squash: Whether to use a tanh function on the continuous output, or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy in continuous output.
        """
        super().__init__(seed, brain, trainer_params, load)
        self.grads = None
        num_layers = trainer_params["num_layers"]
        self.h_size = trainer_params["hidden_units"]
        if num_layers < 1:
            num_layers = 1
        self.num_layers = num_layers
        self.vis_encode_type = EncoderType(
            trainer_params.get("vis_encode_type", "simple")
        )
        self.tanh_squash = tanh_squash
        self.reparameterize = reparameterize
        self.condition_sigma_on_obs = condition_sigma_on_obs

        # Non-exposed parameters; these aren't exposed because they don't have a
        # good explanation and usually shouldn't be touched.
        self.log_std_min = -20
        self.log_std_max = 2

        self.inference_dict: Dict[str, tf.Tensor] = {}
        self.update_dict: Dict[str, tf.Tensor] = {}
        # TF defaults to 32-bit, so we use the same here.
        torch.set_default_tensor_type(torch.DoubleTensor)

        reward_signal_configs = trainer_params["reward_signals"]
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.model = Actor(
            h_size=int(trainer_params["hidden_units"]),
            act_type=ActionType.CONTINUOUS,
            vector_sizes=[brain.vector_observation_space_size],
            act_size=sum(brain.vector_action_space_size),
            normalize=trainer_params["normalize"],
            num_layers=int(trainer_params["num_layers"]),
            m_size=trainer_params["memory_size"],
            use_lstm=self.use_recurrent,
            visual_sizes=brain.camera_resolutions,
            vis_encode_type=EncoderType(
                trainer_params.get("vis_encode_type", "simple")
            ),
        )

        self.critic = Critic(
            h_size=int(trainer_params["hidden_units"]),
            vector_sizes=[brain.vector_observation_space_size],
            normalize=trainer_params["normalize"],
            num_layers=int(trainer_params["num_layers"]),
            m_size=trainer_params["memory_size"],
            use_lstm=self.use_recurrent,
            visual_sizes=brain.camera_resolutions,
            stream_names=list(reward_signal_configs.keys()),
            vis_encode_type=EncoderType(
                trainer_params.get("vis_encode_type", "simple")
            ),
        )

    def split_decision_step(self, decision_requests):
        vec_vis_obs = SplitObservations.from_observations(decision_requests.obs)
        mask = None
        if not self.use_continuous_act:
            mask = np.ones(
                (len(decision_requests), np.sum(self.brain.vector_action_space_size)),
                dtype=np.float32,
            )
            if decision_requests.action_mask is not None:
                mask = 1 - np.concatenate(decision_requests.action_mask, axis=1)
        return vec_vis_obs.vector_observations, vec_vis_obs.visual_observations, mask

    def execute_model(self, vec_obs, vis_obs, masks):
        action_dist = self.model(vec_obs, vis_obs, masks)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value_heads = self.model.get_values(vec_obs, vis_obs)
        return action, log_probs, entropy, value_heads

    @timed
    def evaluate(self, decision_requests: DecisionSteps) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param decision_requests: DecisionStep object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        vec_obs, vis_obs, masks = self.split_decision_step(decision_requests)
        run_out = {}
        action, log_probs, entropy, value_heads = self.execute_model(
            vec_obs, vis_obs, masks
        )
        run_out["action"] = np.array(action.detach())
        run_out["log_probs"] = np.array(log_probs.detach())
        run_out["entropy"] = np.array(entropy.detach())
        run_out["value_heads"] = {
            name: np.array(t.detach()) for name, t in value_heads.items()
        }
        run_out["value"] = np.mean(list(run_out["value_heads"].values()), 0)
        run_out["learning_rate"] = 0.0
        self.model.update_normalization(vec_obs)
        return run_out

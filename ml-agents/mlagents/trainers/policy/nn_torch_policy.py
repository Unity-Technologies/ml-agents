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
    SimpleVisualEncoder,
    ValueHeads,
    Normalizer,
)
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.distributions_torch import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)

EPSILON = 1e-7  # Small value to avoid divide by zero


class Critic(nn.Module):
    def __init__(self, stream_names, hidden_size, encoder, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.stream_names = stream_names
        self.encoder = encoder
        self.value_heads = ValueHeads(stream_names, hidden_size)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return self.value_heads(hidden)


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        h_size,
        vector_sizes,
        visual_sizes,
        act_size,
        normalize,
        num_layers,
        m_size,
        stream_names,
        vis_encode_type,
        act_type,
        use_lstm,
    ):
        super(ActorCriticPolicy, self).__init__()
        self.visual_encoders = []
        self.vector_encoders = []
        self.vector_normalizers = []
        self.act_type = act_type
        self.use_lstm = use_lstm
        self.h_size = h_size
        for vector_size in vector_sizes:
            self.vector_normalizers.append(Normalizer(vector_size))
            self.vector_encoders.append(VectorEncoder(vector_size, h_size, num_layers))
        for visual_size in visual_sizes:
            self.visual_encoders.append(SimpleVisualEncoder(visual_size))

        if use_lstm:
            self.lstm = nn.LSTM(h_size, h_size, 1)

        if self.act_type == ActionType.CONTINUOUS:
            self.distribution = GaussianDistribution(h_size, act_size)
        else:
            self.distribution = MultiCategoricalDistribution(h_size, act_size)

        self.critic = Critic(
            stream_names, h_size, VectorEncoder(vector_sizes[0], h_size, num_layers)
        )
        self.act_size = act_size

    def clear_memory(self, batch_size):
        self.memory = (
            torch.zeros(1, batch_size, self.h_size),
            torch.zeros(1, batch_size, self.h_size),
        )

    def forward(self, vec_inputs, vis_inputs, masks=None):
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

        if self.act_type == ActionType.CONTINUOUS:
            dist = self.distribution(embedding)
        else:
            dist = self.distribution(embedding, masks=masks)
        return dist

    def update_normalization(self, inputs):
        if self.normalize:
            self.normalizer.update(inputs)

    def get_values(self, vec_inputs, vis_inputs):
        if self.normalize:
            vec_inputs = self.normalizer(vec_inputs)
        return self.critic(vec_inputs)


class NNPolicy(TorchPolicy):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        is_training: bool,
        load: bool,
        tanh_squash: bool = False,
        reparameterize: bool = False,
        condition_sigma_on_obs: bool = True,
        create_tf_graph: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous action spaces, as well as recurrent networks.
        :param seed: Random seed.
        :param brain: Assigned BrainParameters object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
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

        self.model = ActorCriticPolicy(
            h_size=int(trainer_params["hidden_units"]),
            act_type=ActionType.CONTINUOUS,
            vector_sizes=[brain.vector_observation_space_size],
            act_size=sum(brain.vector_action_space_size),
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
        :param decision_step: DecisionStep object containing inputs.
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
        self.model.update_normalization(decision_requests.vec_obs)
        return run_out

    # def _create_cc_actor(
    #     self,
    #     encoded: tf.Tensor,
    #     tanh_squash: bool = False,
    #     reparameterize: bool = False,
    #     condition_sigma_on_obs: bool = True,
    # ) -> None:
    #     """
    #     Creates Continuous control actor-critic model.
    #     :param h_size: Size of hidden linear layers.
    #     :param num_layers: Number of hidden linear layers.
    #     :param vis_encode_type: Type of visual encoder to use if visual input.
    #     :param tanh_squash: Whether to use a tanh function, or a clipped output.
    #     :param reparameterize: Whether we are using the resampling trick to update the policy.
    #     """
    #     if self.use_recurrent:
    #         self.memory_in = tf.placeholder(
    #             shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
    #         )
    #         hidden_policy, memory_policy_out = ModelUtils.create_recurrent_encoder(
    #             encoded, self.memory_in, self.sequence_length_ph, name="lstm_policy"
    #         )
    #
    #         self.memory_out = tf.identity(memory_policy_out, name="recurrent_out")
    #     else:
    #         hidden_policy = encoded
    #
    #     with tf.variable_scope("policy"):
    #         distribution = GaussianDistribution(
    #             hidden_policy,
    #             self.act_size,
    #             reparameterize=reparameterize,
    #             tanh_squash=tanh_squash,
    #             condition_sigma=condition_sigma_on_obs,
    #         )
    #
    #     if tanh_squash:
    #         self.output_pre = distribution.sample
    #         self.output = tf.identity(self.output_pre, name="action")
    #     else:
    #         self.output_pre = distribution.sample
    #         # Clip and scale output to ensure actions are always within [-1, 1] range.
    #         output_post = tf.clip_by_value(self.output_pre, -3, 3) / 3
    #         self.output = tf.identity(output_post, name="action")
    #
    #     self.selected_actions = tf.stop_gradient(self.output)
    #
    #     self.all_log_probs = tf.identity(distribution.log_probs, name="action_probs")
    #     self.entropy = distribution.entropy
    #
    #     # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
    #     self.total_log_probs = distribution.total_log_probs
    #
    # def _create_dc_actor(self, encoded: tf.Tensor) -> None:
    #     """
    #     Creates Discrete control actor-critic model.
    #     :param h_size: Size of hidden linear layers.
    #     :param num_layers: Number of hidden linear layers.
    #     :param vis_encode_type: Type of visual encoder to use if visual input.
    #     """
    #     if self.use_recurrent:
    #         self.prev_action = tf.placeholder(
    #             shape=[None, len(self.act_size)], dtype=tf.int32, name="prev_action"
    #         )
    #         prev_action_oh = tf.concat(
    #             [
    #                 tf.one_hot(self.prev_action[:, i], self.act_size[i])
    #                 for i in range(len(self.act_size))
    #             ],
    #             axis=1,
    #         )
    #         hidden_policy = tf.concat([encoded, prev_action_oh], axis=1)
    #
    #         self.memory_in = tf.placeholder(
    #             shape=[None, self.m_size], dtype=tf.float32, name="recurrent_in"
    #         )
    #         hidden_policy, memory_policy_out = ModelUtils.create_recurrent_encoder(
    #             hidden_policy,
    #             self.memory_in,
    #             self.sequence_length_ph,
    #             name="lstm_policy",
    #         )
    #
    #         self.memory_out = tf.identity(memory_policy_out, "recurrent_out")
    #     else:
    #         hidden_policy = encoded
    #
    #     self.action_masks = tf.placeholder(
    #         shape=[None, sum(self.act_size)], dtype=tf.float32, name="action_masks"
    #     )
    #
    #     with tf.variable_scope("policy"):
    #         distribution = MultiCategoricalDistribution(
    #             hidden_policy, self.act_size, self.action_masks
    #         )
    #     # It's important that we are able to feed_dict a value into this tensor to get the
    #     # right one-hot encoding, so we can't do identity on it.
    #     self.output = distribution.sample
    #     self.all_log_probs = tf.identity(distribution.log_probs, name="action")
    #     self.selected_actions = tf.stop_gradient(
    #         distribution.sample_onehot
    #     )  # In discrete, these are onehot
    #     self.entropy = distribution.entropy
    #     self.total_log_probs = distribution.total_log_probs

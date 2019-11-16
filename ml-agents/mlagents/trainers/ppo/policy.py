import logging

import numpy as np
from typing import Any, Dict  # , Optional
import tensorflow as tf
import tensorflow_probability as tfp

from mlagents.envs.timers import timed
from mlagents.envs.brain import BrainInfo, BrainParameters
from mlagents.envs.action_info import ActionInfo
from mlagents.trainers.models import EncoderType  # , LearningRateSchedule

from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)

logger = logging.getLogger("mlagents.trainers")


class VectorEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_layers, **kwargs):
        super(VectorEncoder, self).__init__(**kwargs)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(
                tf.keras.layers.Dense(hidden_size, activation=tf.keras.activations.relu)
            )

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class Critic(tf.keras.layers.Layer):
    def __init__(self, stream_names, encoder, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.stream_names = stream_names
        self.encoder = encoder
        self.value_heads = {}

        for name in stream_names:
            value = tf.keras.layers.Dense(1, name="{}_value".format(name))
            self.value_heads[name] = value

    def call(self, inputs):
        hidden = self.encoder(inputs)
        value_outputs = {}
        for stream_name, value in self.value_heads.items():
            value_outputs[stream_name] = self.value_heads[stream_name](hidden)
        return value_outputs


class GaussianDistribution(tf.keras.layers.Layer):
    def __init__(self, num_outputs, **kwargs):
        super(GaussianDistribution, self).__init__(**kwargs)
        self.mu = tf.keras.layers.Dense(
            num_outputs,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.01),
        )
        self.log_sigma_sq = tf.keras.layers.Dense(
            num_outputs,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.01),
        )
        # self.log_sigma_sq = tf.Variable(
        #     name="log_sig_sq", dtype=tf.float32, initial_value=tf.zeros([num_outputs]), trainable=True
        # )

    def call(self, inputs):
        mu = self.mu(inputs)
        log_sig = self.log_sigma_sq(inputs)
        return tfp.distributions.Normal(loc=mu, scale=tf.sqrt(tf.exp(log_sig)))


class Normalizer(tf.keras.layers.Layer):
    def __init__(self, vec_obs_size, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        print(vec_obs_size)
        self.normalization_steps = tf.Variable(
            name="normalization_steps", trainable=False, dtype=tf.int32, initial_value=1
        )
        self.running_mean = tf.Variable(
            name="running_mean",
            shape=[vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initial_value=tf.zeros([vec_obs_size]),
        )
        self.running_variance = tf.Variable(
            name="running_variance",
            shape=[vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initial_value=tf.ones([vec_obs_size]),
        )

    def call(self, inputs):
        normalized_state = tf.clip_by_value(
            (inputs - self.running_mean)
            / tf.sqrt(
                self.running_variance
                / (tf.cast(self.normalization_steps, tf.float32) + 1)
            ),
            -5,
            5,
            name="normalized_state",
        )
        return normalized_state

    def update(self, vector_input):
        mean_current_observation = tf.cast(
            tf.reduce_mean(vector_input, axis=0), tf.float32
        )
        new_mean = self.running_mean + (
            mean_current_observation - self.running_mean
        ) / tf.cast(tf.add(self.normalization_steps, 1), tf.float32)
        new_variance = self.running_variance + (mean_current_observation - new_mean) * (
            mean_current_observation - self.running_mean
        )
        self.running_mean.assign(new_mean)
        self.running_variance.assign(new_variance)
        self.normalization_steps.assign(self.normalization_steps + 1)


class ActorCriticPolicy(tf.keras.Model):
    def __init__(
        self,
        h_size,
        act_size,
        normalize,
        num_layers,
        m_size,
        stream_names,
        vis_encode_type,
    ):
        super(ActorCriticPolicy, self).__init__()
        self.encoder = VectorEncoder(h_size, num_layers)
        self.distribution = GaussianDistribution(act_size)
        self.critic = Critic(stream_names, VectorEncoder(h_size, num_layers))
        self.act_size = act_size
        self.normalize = normalize
        self.normalizer = None

    def build(self, input_size):
        self.normalizer = Normalizer(input_size[1])

    def call(self, inputs):
        if self.normalize:
            inputs = self.normalizer(inputs)
        _hidden = self.encoder(inputs)
        # epsilon = np.random.normal(size=(input.shape[0], self.act_size))
        dist = self.distribution(_hidden)
        # raw_action = dist.sample()
        # action = tf.clip_by_value(raw_action, -3, 3) / 3
        # log_prob = dist.log_prob(raw_action)
        # entropy = dist.entropy()
        return dist

    @tf.function
    def update_normalization(self, inputs):
        if self.normalize:
            self.normalizer.update(inputs)

    def get_values(self, inputs):
        if self.normalize:
            inputs = self.normalizer(inputs)
        return self.critic(inputs)


class PPOPolicy(object):
    def __init__(
        self,
        seed: int,
        brain: BrainParameters,
        trainer_params: Dict[str, Any],
        is_training: bool,
        load: bool,
    ):
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        # super().__init__(seed, brain, trainer_params)

        reward_signal_configs = trainer_params["reward_signals"]
        self.inference_dict: Dict[str, tf.Tensor] = {}
        self.update_dict: Dict[str, tf.Tensor] = {}
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.create_model(
            brain, trainer_params, reward_signal_configs, is_training, load, seed
        )
        self.brain = brain
        self.trainer_params = trainer_params
        self.optimizer = tf.keras.optimizers.Adam(
            lr=self.trainer_params["learning_rate"]
        )
        self.sequence_length = (
            1
            if not self.trainer_params["use_recurrent"]
            else self.trainer_params["sequence_length"]
        )
        self.global_step = tf.Variable(0)
        self.create_reward_signals(reward_signal_configs)

    def create_model(
        self, brain, trainer_params, reward_signal_configs, is_training, load, seed
    ):
        """
        Create PPO model
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param reward_signal_configs: Reward signal config
        :param seed: Random seed.
        """
        self.model = ActorCriticPolicy(
            h_size=int(trainer_params["hidden_units"]),
            act_size=sum(brain.vector_action_space_size),
            normalize=trainer_params["normalize"],
            num_layers=int(trainer_params["num_layers"]),
            m_size=trainer_params["memory_size"],
            stream_names=list(reward_signal_configs.keys()),
            vis_encode_type=EncoderType(
                trainer_params.get("vis_encode_type", "simple")
            ),
        )

    def ppo_value_loss(self, values, old_values, returns):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value_heads: Value estimate tensors from each value stream
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """

        decay_epsilon = self.trainer_params["epsilon"]

        value_losses = []
        for name, head in values.items():
            clipped_value_estimate = old_values[name] + tf.clip_by_value(
                tf.reduce_sum(head, axis=1) - old_values[name],
                -decay_epsilon,
                decay_epsilon,
            )
            v_opt_a = tf.math.squared_difference(
                returns[name], tf.reduce_sum(head, axis=1)
            )
            v_opt_b = tf.math.squared_difference(returns[name], clipped_value_estimate)
            value_loss = tf.reduce_mean(tf.maximum(v_opt_a, v_opt_b))
            value_losses.append(value_loss)
        value_loss = tf.reduce_mean(value_losses)
        return value_loss

    def ppo_policy_loss(self, advantages, probs, old_probs, masks, epsilon):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value_heads: Value estimate tensors from each value stream
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """
        advantage = tf.expand_dims(advantages, -1)

        decay_epsilon = self.trainer_params["epsilon"]

        r_theta = tf.exp(probs - old_probs)
        p_opt_a = r_theta * advantage
        p_opt_b = (
            tf.clip_by_value(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon)
            * advantage
        )
        # print(tf.reduce_mean(p_opt_a), tf.reduce_mean(p_opt_b))
        policy_loss = -tf.reduce_mean(tf.minimum(p_opt_a, p_opt_b))
        # For cleaner stats reporting
        # abs_policy_loss = tf.abs(policy_loss)
        return policy_loss

    def create_reward_signals(self, reward_signal_configs):
        """
        Create reward signals
        :param reward_signal_configs: Reward signal config.
        """
        self.reward_signals = {}
        # with self.graph.as_default():
        # Create reward signals
        for reward_signal, config in reward_signal_configs.items():
            self.reward_signals[reward_signal] = create_reward_signal(
                self, self.model, reward_signal, config
            )
            self.update_dict.update(self.reward_signals[reward_signal].update_dict)

    @tf.function
    def execute_model(self, observations):
        action_dist = self.model(observations)
        action = action_dist.sample()
        log_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value_heads = self.model.get_values(observations)
        return action, log_probs, entropy, value_heads

    @timed
    def evaluate(self, brain_info):
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """

        run_out = {}
        action, log_probs, entropy, value_heads = self.execute_model(
            brain_info.vector_observations
        )
        run_out["action"] = np.array(action)
        run_out["log_probs"] = np.array(log_probs)
        run_out["entropy"] = np.array(entropy)
        run_out["value_heads"] = {name: np.array(t) for name, t in value_heads.items()}
        run_out["value"] = np.mean(list(run_out["value_heads"].values()), 0)
        run_out["learning_rate"] = 0.0
        self.model.update_normalization(brain_info.vector_observations)
        return run_out

    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(brain_info.agents) == 0:
            return ActionInfo([], [], None)
        run_out = self.evaluate(brain_info)  # pylint: disable=assignment-from-no-return
        return ActionInfo(
            action=run_out.get("action"), value=run_out.get("value"), outputs=run_out
        )

    @timed
    def update(self, mini_batch, num_sequences):
        """
        Performs update on model.
        :param mini_batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        with tf.GradientTape() as tape:
            returns = {}
            old_values = {}
            for name in self.reward_signals:
                returns[name] = mini_batch["{}_returns".format(name)]
                old_values[name] = mini_batch["{}_value_estimates".format(name)]

            obs = np.array(mini_batch["vector_obs"])
            values = self.model.get_values(obs)
            dist = self.model(obs)
            probs = dist.log_prob(np.array(mini_batch["actions"]))
            entropy = dist.entropy()
            value_loss = self.ppo_value_loss(values, old_values, returns)
            policy_loss = self.ppo_policy_loss(
                np.array(mini_batch["advantages"]),
                probs,
                np.array(mini_batch["action_probs"]),
                np.array(mini_batch["masks"], dtype=np.uint32),
                1e-3,
            )
            loss = (
                policy_loss
                + 0.5 * value_loss
                - self.trainer_params["beta"] * tf.reduce_mean(entropy)
            )
        grads = tape.gradient(loss, self.model.trainable_weights)
        # for grad,weight in zip(grads, self.model.trainable_weights):
        #     if "critic/" in weight.name:
        #         print(grad,weight.name)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        update_stats = {}
        update_stats["Losses/Policy Loss"] = abs(policy_loss)
        update_stats["Losses/Value Loss"] = value_loss
        # for stat_name, update_name in stats_needed.items():
        #     update_stats[stat_name] = update_vals[update_name]
        return update_stats

    def get_value_estimates(
        self, brain_info: BrainInfo, idx: int, done: bool
    ) -> Dict[str, float]:
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :param done: Whether or not this is the last element of the episode, in which case the value estimate will be 0.
        :return: The value estimate dictionary with key being the name of the reward signal and the value the
        corresponding value estimate.
        """
        value_estimates = self.model.get_values(
            np.expand_dims(brain_info.vector_observations[idx], 0)
        )

        value_estimates = {k: float(v) for k, v in value_estimates.items()}

        # If we're done, reassign all of the value estimates that need terminal states.
        if done:
            for k in value_estimates:
                if self.reward_signals[k].use_terminal_states:
                    value_estimates[k] = 0.0

        return value_estimates

    @property
    def vis_obs_size(self):
        return self.brain.number_visual_observations

    @property
    def vec_obs_size(self):
        return self.brain.vector_observation_space_size

    @property
    def use_vis_obs(self):
        return self.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.vec_obs_size > 0

    @property
    def use_recurrent(self):
        return False

    @property
    def use_continuous_act(self):
        return True

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        step = self.global_step.numpy()
        return step

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        self.global_step.assign(self.global_step + n_steps)
        return self.get_current_step()

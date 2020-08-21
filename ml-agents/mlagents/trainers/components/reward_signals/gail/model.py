from typing import Optional, Tuple

from mlagents.tf_utils import tf

from mlagents.trainers.policy.tf_policy import TFPolicy
from mlagents.trainers.tf.models import ModelUtils

EPSILON = 1e-7


class GAILModel:
    def __init__(
        self,
        policy: TFPolicy,
        h_size: int = 128,
        learning_rate: float = 3e-4,
        encoding_size: int = 64,
        use_actions: bool = False,
        use_vail: bool = False,
        gradient_penalty_weight: float = 10.0,
    ):
        """
        The initializer for the GAIL reward generator.
        https://arxiv.org/abs/1606.03476
        :param policy_model: The policy of the learning algorithm
        :param h_size: Size of the hidden layer for the discriminator
        :param learning_rate: The learning Rate for the discriminator
        :param encoding_size: The encoding size for the encoder
        :param use_actions: Whether or not to use actions to discriminate
        :param use_vail: Whether or not to use a variational bottleneck for the
        discriminator. See https://arxiv.org/abs/1810.00821.
        """
        self.h_size = h_size
        self.z_size = 128
        self.alpha = 0.0005
        self.mutual_information = 0.5
        self.policy = policy
        self.encoding_size = encoding_size
        self.gradient_penalty_weight = gradient_penalty_weight
        self.use_vail = use_vail
        self.use_actions = use_actions  # True # Not using actions

        self.noise: Optional[tf.Tensor] = None
        self.z: Optional[tf.Tensor] = None

        self.make_inputs()
        self.create_network()
        self.create_loss(learning_rate)
        if self.use_vail:
            self.make_beta_update()

    def make_beta_update(self) -> None:
        """
        Creates the beta parameter and its updater for GAIL
        """

        new_beta = tf.maximum(
            self.beta + self.alpha * (self.kl_loss - self.mutual_information), EPSILON
        )
        with tf.control_dependencies([self.update_batch]):
            self.update_beta = tf.assign(self.beta, new_beta)

    def make_inputs(self) -> None:
        """
        Creates the input layers for the discriminator
        """
        self.done_expert_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.done_policy_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.done_expert = tf.expand_dims(self.done_expert_holder, -1)
        self.done_policy = tf.expand_dims(self.done_policy_holder, -1)

        if self.policy.behavior_spec.is_action_continuous():
            action_length = self.policy.act_size[0]
            self.action_in_expert = tf.placeholder(
                shape=[None, action_length], dtype=tf.float32
            )
            self.expert_action = tf.identity(self.action_in_expert)
        else:
            action_length = len(self.policy.act_size)
            self.action_in_expert = tf.placeholder(
                shape=[None, action_length], dtype=tf.int32
            )
            self.expert_action = tf.concat(
                [
                    tf.one_hot(self.action_in_expert[:, i], act_size)
                    for i, act_size in enumerate(self.policy.act_size)
                ],
                axis=1,
            )

        encoded_policy_list = []
        encoded_expert_list = []

        (
            self.obs_in_expert,
            self.expert_visual_in,
        ) = ModelUtils.create_input_placeholders(
            self.policy.behavior_spec.observation_shapes, "gail_"
        )

        if self.policy.vec_obs_size > 0:
            if self.policy.normalize:
                encoded_expert_list.append(
                    ModelUtils.normalize_vector_obs(
                        self.obs_in_expert,
                        self.policy.running_mean,
                        self.policy.running_variance,
                        self.policy.normalization_steps,
                    )
                )
                encoded_policy_list.append(self.policy.processed_vector_in)
            else:
                encoded_expert_list.append(self.obs_in_expert)
                encoded_policy_list.append(self.policy.vector_in)

        if self.expert_visual_in:
            visual_policy_encoders = []
            visual_expert_encoders = []
            for i, (vis_in, exp_vis_in) in enumerate(
                zip(self.policy.visual_in, self.expert_visual_in)
            ):
                encoded_policy_visual = ModelUtils.create_visual_observation_encoder(
                    vis_in,
                    self.encoding_size,
                    ModelUtils.swish,
                    1,
                    f"gail_stream_{i}_visual_obs_encoder",
                    False,
                )

                encoded_expert_visual = ModelUtils.create_visual_observation_encoder(
                    exp_vis_in,
                    self.encoding_size,
                    ModelUtils.swish,
                    1,
                    f"gail_stream_{i}_visual_obs_encoder",
                    True,
                )
                visual_policy_encoders.append(encoded_policy_visual)
                visual_expert_encoders.append(encoded_expert_visual)
            hidden_policy_visual = tf.concat(visual_policy_encoders, axis=1)
            hidden_expert_visual = tf.concat(visual_expert_encoders, axis=1)
            encoded_policy_list.append(hidden_policy_visual)
            encoded_expert_list.append(hidden_expert_visual)

        self.encoded_expert = tf.concat(encoded_expert_list, axis=1)
        self.encoded_policy = tf.concat(encoded_policy_list, axis=1)

    def create_encoder(
        self, state_in: tf.Tensor, action_in: tf.Tensor, done_in: tf.Tensor, reuse: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Creates the encoder for the discriminator
        :param state_in: The encoded observation input
        :param action_in: The action input
        :param done_in: The done flags input
        :param reuse: If true, the weights will be shared with the previous encoder created
        """
        with tf.variable_scope("GAIL_model"):
            if self.use_actions:
                concat_input = tf.concat([state_in, action_in, done_in], axis=1)
            else:
                concat_input = state_in

            hidden_1 = tf.layers.dense(
                concat_input,
                self.h_size,
                activation=ModelUtils.swish,
                name="gail_d_hidden_1",
                reuse=reuse,
            )

            hidden_2 = tf.layers.dense(
                hidden_1,
                self.h_size,
                activation=ModelUtils.swish,
                name="gail_d_hidden_2",
                reuse=reuse,
            )

            z_mean = None
            if self.use_vail:
                # Latent representation
                z_mean = tf.layers.dense(
                    hidden_2,
                    self.z_size,
                    reuse=reuse,
                    name="gail_z_mean",
                    kernel_initializer=ModelUtils.scaled_init(0.01),
                )

                self.noise = tf.random_normal(tf.shape(z_mean), dtype=tf.float32)

                # Sampled latent code
                self.z = z_mean + self.z_sigma * self.noise * self.use_noise
                estimate_input = self.z
            else:
                estimate_input = hidden_2

            estimate = tf.layers.dense(
                estimate_input,
                1,
                activation=tf.nn.sigmoid,
                name="gail_d_estimate",
                reuse=reuse,
            )
            return estimate, z_mean, concat_input

    def create_network(self) -> None:
        """
        Helper for creating the intrinsic reward nodes
        """
        if self.use_vail:
            self.z_sigma = tf.get_variable(
                "gail_sigma_vail",
                self.z_size,
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
            )
            self.z_sigma_sq = self.z_sigma * self.z_sigma
            self.z_log_sigma_sq = tf.log(self.z_sigma_sq + EPSILON)
            self.use_noise = tf.placeholder(
                shape=[1], dtype=tf.float32, name="gail_NoiseLevel"
            )
        self.expert_estimate, self.z_mean_expert, _ = self.create_encoder(
            self.encoded_expert, self.expert_action, self.done_expert, reuse=False
        )
        self.policy_estimate, self.z_mean_policy, _ = self.create_encoder(
            self.encoded_policy,
            self.policy.selected_actions,
            self.done_policy,
            reuse=True,
        )
        self.mean_policy_estimate = tf.reduce_mean(self.policy_estimate)
        self.mean_expert_estimate = tf.reduce_mean(self.expert_estimate)
        self.discriminator_score = tf.reshape(
            self.policy_estimate, [-1], name="gail_reward"
        )
        self.intrinsic_reward = -tf.log(1.0 - self.discriminator_score + EPSILON)

    def create_gradient_magnitude(self) -> tf.Tensor:
        """
        Gradient penalty from https://arxiv.org/pdf/1704.00028. Adds stability esp.
        for off-policy. Compute gradients w.r.t randomly interpolated input.
        """
        expert = [self.encoded_expert, self.expert_action, self.done_expert]
        policy = [self.encoded_policy, self.policy.selected_actions, self.done_policy]
        interp = []
        for _expert_in, _policy_in in zip(expert, policy):
            alpha = tf.random_uniform(tf.shape(_expert_in))
            interp.append(alpha * _expert_in + (1 - alpha) * _policy_in)

        grad_estimate, _, grad_input = self.create_encoder(
            interp[0], interp[1], interp[2], reuse=True
        )

        grad = tf.gradients(grad_estimate, [grad_input])[0]

        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=-1) + EPSILON)
        gradient_mag = tf.reduce_mean(tf.pow(safe_norm - 1, 2))

        return gradient_mag

    def create_loss(self, learning_rate: float) -> None:
        """
        Creates the loss and update nodes for the GAIL reward generator
        :param learning_rate: The learning rate for the optimizer
        """
        self.mean_expert_estimate = tf.reduce_mean(self.expert_estimate)
        self.mean_policy_estimate = tf.reduce_mean(self.policy_estimate)

        if self.use_vail:
            self.beta = tf.get_variable(
                "gail_beta",
                [],
                trainable=False,
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
            )

        self.discriminator_loss = -tf.reduce_mean(
            tf.log(self.expert_estimate + EPSILON)
            + tf.log(1.0 - self.policy_estimate + EPSILON)
        )

        if self.use_vail:
            # KL divergence loss (encourage latent representation to be normal)
            self.kl_loss = tf.reduce_mean(
                -tf.reduce_sum(
                    1
                    + self.z_log_sigma_sq
                    - 0.5 * tf.square(self.z_mean_expert)
                    - 0.5 * tf.square(self.z_mean_policy)
                    - tf.exp(self.z_log_sigma_sq),
                    1,
                )
            )
            self.loss = (
                self.beta * (self.kl_loss - self.mutual_information)
                + self.discriminator_loss
            )
        else:
            self.loss = self.discriminator_loss

        if self.gradient_penalty_weight > 0.0:
            self.loss += self.gradient_penalty_weight * self.create_gradient_magnitude()

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.minimize(self.loss)
